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

        const PenguinV_Image::Image & image() const
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
            const double xFactor = static_cast<double>(clientRoi.right - clientRoi.left) / _window->_image.width();
            const double yFactor = static_cast<double>(clientRoi.bottom - clientRoi.top) / _window->_image.height();

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
            , className( L"UiWindowWin" )
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
        const std::wstring className;
    };
}

UiWindowWin::UiWindowWin( const PenguinV_Image::Image & image, const std::string & title )
    : UiWindow( PenguinV_Image::Image(), title ) // we pass an empty image into base class
    , _bmpInfo( nullptr )
{
    static const UiWindowWinRegistrator registrator; // we need to register only once hence static variable
    if ( !registrator.registered )
        throw imageException( "Unable to create Windows API class" );

    UiWindowWin::setImage( image );

    const std::wstring titleWChar = std::wstring( title.begin(), title.end() );
    _window = CreateWindowEx( 0, registrator.className.data(), titleWChar.data(), WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
                              CW_USEDEFAULT, CW_USEDEFAULT, 0, 0, GetModuleHandle( 0 ), this ) ;

    if ( _window == nullptr )
        throw imageException( "Unable to create Windows API window" );

    RECT clientRoi;
    GetClientRect( _window, &clientRoi );
    RECT windowRoi;
    GetWindowRect( _window, &windowRoi );

    // Resize window to fit image 1 to 1
    MoveWindow( _window, windowRoi.left, windowRoi.top,
                windowRoi.right - windowRoi.left - (clientRoi.right - clientRoi.left) + image.width(),
                windowRoi.bottom - windowRoi.top - (clientRoi.bottom - clientRoi.top) + image.height(), FALSE );
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

void UiWindowWin::setImage( const PenguinV_Image::Image & image )
{
    _free();

    // We need to set image upside-down to show correctly in UI window as well as make 4 byte row length for bitmap
    _image.setAlignment( 4u );
    _image.setColorCount( image.colorCount() );
    _image.resize( image.width(), image.height() );

    const uint32_t rowSizeIn  = image.rowSize();
    const uint32_t rowSizeOut = _image.rowSize();
    const uint32_t width      = image.width() * image.colorCount();

    const uint8_t * inY     = image.data() + image.rowSize() * (image.height() - 1);
    uint8_t       * outY    = _image.data();
    const uint8_t * outYEnd = outY + image.height() * rowSizeOut;

    for ( ; outY != outYEnd; outY += rowSizeOut, inY -= rowSizeIn )
        memcpy( outY, inY, sizeof( uint8_t ) * width );

    const bool rgbImage = (image.colorCount() != 1u);
    const DWORD bmpInfoSize = sizeof( BITMAPINFOHEADER ) + (rgbImage ? 1 : 256) * sizeof( RGBQUAD );

    _bmpInfo = reinterpret_cast<BITMAPINFO*>(malloc( bmpInfoSize ));
    _bmpInfo->bmiHeader.biSize          = sizeof( BITMAPINFOHEADER );
    _bmpInfo->bmiHeader.biWidth         = image.width();
    _bmpInfo->bmiHeader.biHeight        = image.height();
    _bmpInfo->bmiHeader.biPlanes        = 1;
    _bmpInfo->bmiHeader.biBitCount      = image.colorCount() * 8;
    _bmpInfo->bmiHeader.biCompression   = BI_RGB;
    _bmpInfo->bmiHeader.biSizeImage     = image.rowSize() * image.height();
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

void UiWindowWin::_free()
{
    if ( _bmpInfo != nullptr ) {
        free( _bmpInfo );
        _bmpInfo = nullptr;
    }
}

#endif
