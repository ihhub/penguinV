#include "win_ui.h"

#ifdef _WIN32 // Windows

#include <iostream>
#include <string>

namespace WindowsUi
{
    // This is a wrapper class which was made to avoid UiWindowWin class member exposure to public
    class UiWindowWinInfo
    {
    public:
        explicit UiWindowWinInfo( const UiWindowWin * window )
            : _window( window )
        {
        }

        const penguinV::Image & image() const
        {
            return _window->_image;
        }

        const BITMAPINFO * bitmapInfo() const
        {
            return _window->_bmpInfo;
        }

        bool valid() const
        {
            return (_window != nullptr) && !_window->_image.empty() && (_window->_bmpInfo != nullptr);
        }

        void paint( HDC hdc, RECT clientRoi ) const
        {
            const double xFactor = static_cast<double>( clientRoi.right - clientRoi.left ) / _window->_image.width() * _window->_scaleFactor;
            const double yFactor = static_cast<double>( clientRoi.bottom - clientRoi.top ) / _window->_image.height() * _window->_scaleFactor;

            const int minPointFactor = static_cast<int>(xFactor < yFactor ? xFactor : yFactor);
            const int pointMultiplicator = minPointFactor > 1 ? minPointFactor / 2 : 1;

            for ( std::vector < UiWindowWin::PointToDraw >::const_iterator point = _window->_point.cbegin(); point != _window->_point.cend(); ++point ) {
                const int x = static_cast<int>(point->point.x * xFactor);
                const int y = static_cast<int>(point->point.y * yFactor);

                HPEN hPen = CreatePen( PS_SOLID, 1, RGB( point->color.red, point->color.green, point->color.blue ) );
                HGDIOBJ hOldPen = SelectObject( hdc, hPen );
                MoveToEx( hdc, x - pointMultiplicator, y - pointMultiplicator, NULL );
                LineTo  ( hdc, x + pointMultiplicator, y + pointMultiplicator );
                MoveToEx( hdc, x + pointMultiplicator, y - pointMultiplicator, NULL );
                LineTo  ( hdc, x - pointMultiplicator, y + pointMultiplicator );
                SelectObject( hdc, hOldPen );
                DeleteObject( hPen );
            }

            for ( std::vector < UiWindowWin::LineToDraw >::const_iterator line = _window->_line.cbegin(); line != _window->_line.cend(); ++line ) {
                const int xStart = static_cast<int>(line->start.x * xFactor);
                const int yStart = static_cast<int>(line->start.y * yFactor);
                const int xEnd   = static_cast<int>(line->end.x * xFactor);
                const int yEnd   = static_cast<int>(line->end.y * yFactor);

                HPEN hPen = CreatePen( PS_SOLID, 1, RGB( line->color.red, line->color.green, line->color.blue ) );
                HGDIOBJ hOldPen = SelectObject( hdc, hPen );
                MoveToEx( hdc, xStart, yStart, NULL );
                LineTo( hdc, xEnd, yEnd );
                SelectObject( hdc, hOldPen );
                DeleteObject( hPen );
            }

            for ( std::vector < UiWindowWin::EllipseToDraw >::const_iterator ellipse = _window->_ellipse.cbegin(); ellipse != _window->_ellipse.cend(); ++ellipse ) {
                const int left = static_cast<int>( ellipse->left * xFactor );
                const int top = static_cast<int>( ellipse->top * yFactor );
                const int right = static_cast<int>( ellipse->right * xFactor );
                const int bottom = static_cast<int>( ellipse->bottom * yFactor );

                HPEN hPen = CreatePen( PS_SOLID, 1, RGB( ellipse->color.red, ellipse->color.green, ellipse->color.blue ) );
                HGDIOBJ hOldPen = SelectObject( hdc, hPen );
                Arc( hdc, left, top, right, bottom, 0, 0, 0, 0 );
                SelectObject( hdc, hOldPen );
                DeleteObject( hPen );
            }
        }

    private:
        const UiWindowWin * _window;
    };
}

namespace
{
    LRESULT __stdcall WindowProcedure( HWND window, unsigned int msg, WPARAM wp, LPARAM lp )
    {
        switch ( msg ) {
            case WM_CREATE:
                // register owner class object pointer
                SetWindowLongPtr( window, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(reinterpret_cast<CREATESTRUCT*>(lp)->lpCreateParams) );
                break;
            case WM_PAINT:
            {
                PAINTSTRUCT paintStructure;
                HDC handleToDisplayDevice = BeginPaint( window, &paintStructure );

                const WindowsUi::UiWindowWinInfo info( reinterpret_cast<UiWindowWin*>(GetWindowLongPtr( window, GWLP_USERDATA )) );
                if ( info.valid() ) { // make sure that we draw a valid image
                    RECT clientRoi;
                    GetClientRect( window, &clientRoi );
                    const DWORD result = StretchDIBits( handleToDisplayDevice, 0, 0, clientRoi.right, clientRoi.bottom,
                                                        0, 0, info.image().width(), info.image().height(),
                                                        info.image().data(), info.bitmapInfo(), DIB_RGB_COLORS, SRCCOPY );
                    if ( result != info.image().height() )
                        DebugBreak(); // drawing failed

                    info.paint( handleToDisplayDevice, clientRoi );
                }
                EndPaint( window, &paintStructure );
            }
            break;
            case WM_DESTROY:
                PostQuitMessage( 0 ) ;
            default:
                return DefWindowProc( window, msg, wp, lp );
        }

        return 0;
    }

    struct UiWindowWinRegistrator
    {
        UiWindowWinRegistrator()
            : registered( 0 )
#ifdef UNICODE
            , className( L"UiWindowWin" )
#else
            , className( "UiWindowWin" )
#endif
        {
            WNDCLASSEX wcex;
            wcex.cbSize        = sizeof( WNDCLASSEX );
            wcex.style         = CS_HREDRAW | CS_VREDRAW;
            wcex.lpfnWndProc   = WindowProcedure;
            wcex.cbClsExtra    = 0;
            wcex.cbWndExtra    = 0;
            wcex.hInstance     = 0;
            wcex.hIcon         = 0;
            wcex.hCursor       = LoadCursor( NULL, IDC_CROSS );
            wcex.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW);
            wcex.lpszMenuName  = NULL;
            wcex.lpszClassName = className.data();
            wcex.hIconSm       = 0;

            registered = RegisterClassEx( &wcex );
        }
        WORD registered;
#ifdef UNICODE
        const std::wstring className;
#else
        const std::string className;
#endif
    };
}

UiWindowWin::UiWindowWin( const penguinV::Image & image, const std::string & title )
    : UiWindow( penguinV::Image(), title ) // we pass an empty image into base class
    , _bmpInfo( nullptr )
    , _scaleFactor( 1.0 )
{
    static const UiWindowWinRegistrator registrator; // we need to register only once hence static variable
    if ( !registrator.registered )
        throw imageException( "Unable to create Windows API class" );

#ifdef UNICODE
    const std::wstring titleName = std::wstring( title.begin(), title.end() );
#else
    const std::string & titleName = title;
#endif
    _window = CreateWindowEx( 0, registrator.className.data(), titleName.data(), WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
                              CW_USEDEFAULT, CW_USEDEFAULT, 0, 0, GetModuleHandle( 0 ), this ) ;

    if ( _window == nullptr )
        throw imageException( "Unable to create Windows API window" );

    UiWindowWin::setImage( image );
}

UiWindowWin::~UiWindowWin()
{
    _free();
}

void UiWindowWin::_display()
{
    if ( !_shown ) // we don't need to display anything
        return;

    ShowWindow( _window, SW_SHOWDEFAULT );
    UpdateWindow( _window );

    MSG windowMessage;
    while ( GetMessage( &windowMessage, NULL, 0, 0 ) ) {
        TranslateMessage( &windowMessage );
        DispatchMessage( &windowMessage );
    }
}

void UiWindowWin::setImage( const penguinV::Image & image )
{
    _free();

    // We need to set image upside-down to show correctly in UI window as well as make 4 byte row length for bitmap
    _image.setAlignment( 4u );
    _image.setColorCount( image.colorCount() );

    const int maxWindowWidth = GetSystemMetrics( SM_CXSCREEN ) * 95 / 100;
    const int maxWindowHeight = GetSystemMetrics( SM_CYSCREEN ) * 95 / 100;
    if ( maxWindowWidth > 0 && static_cast<uint32_t>( maxWindowWidth ) < image.width() ||
         maxWindowHeight > 0 && static_cast<uint32_t>( maxWindowHeight ) < image.height() ) {
        const double scaleX = static_cast<double>( maxWindowWidth ) / image.width();
        const double scaleY = static_cast<double>( maxWindowHeight ) / image.height();

        _scaleFactor = scaleX < scaleY ? scaleX : scaleY;
        _image.resize( static_cast<uint32_t>( _scaleFactor * image.width() ), static_cast<uint32_t>( _scaleFactor * image.height() ) );

        const uint8_t colorCount = image.colorCount();

        const uint32_t widthIn = image.width();
        uint32_t widthOut = _image.width();

        const uint32_t heightIn = image.height();
        const uint32_t heightOut = _image.height();

        const uint32_t rowSizeIn  = image.rowSize();
        const uint32_t rowSizeOut = _image.rowSize();

        const uint8_t * inY  = image.data();
        uint8_t       * outY = _image.data();
        const uint8_t * outYEnd = outY + heightOut * rowSizeOut;

        uint32_t idY = 0;

        // Precalculation of X position
        std::vector < uint32_t > positionX( widthOut );
        for ( uint32_t x = 0; x < widthOut; ++x )
            positionX[x] = ( x * widthIn / widthOut ) * colorCount;

        widthOut *= colorCount;

        const size_t pixelSize = sizeof( uint8_t ) * colorCount;

        for ( ; outY != outYEnd; outY += rowSizeOut, ++idY ) {
            uint8_t       * outX = outY;
            const uint8_t * outXEnd = outX + widthOut;

            const uint8_t * inX  = inY + ( heightIn - 1 - ( idY * heightIn / heightOut ) ) * rowSizeIn;
            const uint32_t * idX = positionX.data();

            for ( ; outX != outXEnd; outX += colorCount, ++idX )
                memcpy( outX, inX + (*idX), pixelSize );
        }
    }
    else {
        _scaleFactor = 1.0;

        _image.resize( image.width(), image.height() );

        const uint32_t rowSizeIn  = image.rowSize();
        const uint32_t rowSizeOut = _image.rowSize();
        const uint32_t width      = image.width() * image.colorCount();

        const uint8_t * inY     = image.data() + image.rowSize() * (image.height() - 1);
        uint8_t       * outY    = _image.data();
        const uint8_t * outYEnd = outY + image.height() * rowSizeOut;

        for ( ; outY != outYEnd; outY += rowSizeOut, inY -= rowSizeIn )
            memcpy( outY, inY, sizeof( uint8_t ) * width );
    }

    const bool rgbImage = ( _image.colorCount() != 1u );
    const DWORD bmpInfoSize = sizeof( BITMAPINFOHEADER ) + (rgbImage ? 1 : 256) * sizeof( RGBQUAD );

    _bmpInfo = reinterpret_cast<BITMAPINFO*>(malloc( bmpInfoSize ));
    _bmpInfo->bmiHeader.biSize          = sizeof( BITMAPINFOHEADER );
    _bmpInfo->bmiHeader.biWidth         = _image.width();
    _bmpInfo->bmiHeader.biHeight        = _image.height();
    _bmpInfo->bmiHeader.biPlanes        = 1;
    _bmpInfo->bmiHeader.biBitCount      = _image.colorCount() * 8;
    _bmpInfo->bmiHeader.biCompression   = BI_RGB;
    _bmpInfo->bmiHeader.biSizeImage     = _image.rowSize() * _image.height();
    _bmpInfo->bmiHeader.biXPelsPerMeter = 0;
    _bmpInfo->bmiHeader.biYPelsPerMeter = 0;
    _bmpInfo->bmiHeader.biClrUsed       = rgbImage ? 0 : 256;
    _bmpInfo->bmiHeader.biClrImportant  = 0;

    if ( !rgbImage ) {
        for ( uint8_t i = 0u; ; ++i ) {
            _bmpInfo->bmiColors[i].rgbRed   = i;
            _bmpInfo->bmiColors[i].rgbGreen = i;
            _bmpInfo->bmiColors[i].rgbBlue  = i;
            _bmpInfo->bmiColors[i].rgbReserved = 0;

            if ( i == 255u ) // to avoid variable overflow
                break;
        }
    }

    RECT clientRoi;
    GetClientRect( _window, &clientRoi );
    RECT windowRoi;
    GetWindowRect( _window, &windowRoi );

    // Resize window to fit image 1 to 1
    MoveWindow( _window, windowRoi.left, windowRoi.top,
                windowRoi.right - windowRoi.left - (clientRoi.right - clientRoi.left) + _image.width(),
                windowRoi.bottom - windowRoi.top - (clientRoi.bottom - clientRoi.top) + _image.height(), FALSE );
}

void UiWindowWin::drawPoint( const Point2d & point, const PaintColor & color )
{
    _point.emplace_back( point, color );

    _display();
}

void UiWindowWin::drawLine( const Point2d & start, const Point2d & end, const PaintColor & color )
{
    _line.emplace_back( start, end, color );

    _display();
}

void UiWindowWin::drawEllipse( const Point2d & center, double xRadius, double yRadius, const PaintColor & color )
{
    _ellipse.emplace_back( center.x - xRadius, center.y - yRadius, center.x + xRadius, center.y + yRadius, color );

    _display();
}

void UiWindowWin::drawRectangle( const Point2d & topLeftCorner, double width, double height, const PaintColor & color )
{
    const Point2d topRightCorner( topLeftCorner.x + width, topLeftCorner.y );
    const Point2d bottomLeftCorner( topLeftCorner.x, topLeftCorner.y + height );
    const Point2d bottomRightCorner( topLeftCorner.x + width, topLeftCorner.y + height );

    _line.emplace_back( topLeftCorner, topRightCorner, color );
    _line.emplace_back( topLeftCorner, bottomLeftCorner, color );
    _line.emplace_back( topRightCorner, bottomRightCorner, color );
    _line.emplace_back( bottomLeftCorner, bottomRightCorner, color );

    _display();
}

void UiWindowWin::_free()
{
    if ( _bmpInfo != nullptr ) {
        free( _bmpInfo );
        _bmpInfo = nullptr;
    }
}

#endif
