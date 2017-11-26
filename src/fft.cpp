#include "fft.h"
#include "image_function.h"
#include "image_exception.h"

namespace FFT
{
    ComplexData::ComplexData()
        : _data  ( NULL )
        , _width ( 0 )
        , _height( 0 )
    {
    }

    ComplexData::ComplexData( const Bitmap_Image::Image & image )
        : _data  ( NULL )
        , _width ( 0 )
        , _height( 0 )
    {
        set( image );
    }

    ComplexData::ComplexData( const ComplexData & data )
        : _data  ( NULL )
        , _width ( 0 )
        , _height( 0 )
    {
        _copy( data );
    }

    ComplexData::ComplexData( ComplexData && data )
        : _data  ( NULL )
        , _width ( 0 )
        , _height( 0 )
    {
        _swap( data );
    }

    ComplexData & ComplexData::operator=( const ComplexData & data )
    {
        _copy( data );

        return *this;
    }

    ComplexData & ComplexData::operator=( ComplexData && data )
    {
        _swap( data );

        return *this;
    }

    ComplexData::~ComplexData()
    {
        _clean();
    }

    void ComplexData::set( const Bitmap_Image::Image & image )
    {
        if( image.empty() || image.colorCount() != 1u )
            throw imageException( "Failed to allocate complex data for empty or coloured image" );

        _clean();

        const uint32_t size = image.width() * image.height();

        _data = reinterpret_cast<kiss_fft_cpx *>( malloc( size * sizeof(kiss_fft_cpx) ) );

        _width  = image.width();
        _height = image.height();

        // Copy data from input image to FFT array
        const uint32_t rowSize = image.rowSize();

        const uint8_t * inY  = image.data();
        kiss_fft_cpx  * out = _data;

        const uint8_t * inYEnd = inY + _height * rowSize;

        for( ; inY != inYEnd; inY += rowSize ) {
            const uint8_t * inX  = inY;
            const uint8_t * inXEnd = inX + _width;

            for( ; inX != inXEnd; ++inX, ++out ) {
                out->r = *inX;
                out->i = 0;
            }
        }
    }

    void ComplexData::set( const std::vector<float> & data )
    {
        if( data.empty() || _width == 0 || _height == 0 || data.size() != _width * _height )
            throw imageException( "Failed to allocate complex data for empty or coloured image" );

        const float * in = data.data();
        kiss_fft_cpx * out = _data;
        const kiss_fft_cpx * outEnd = out + _width * _height;

        for( ; out != outEnd; ++in, ++out ) {
            out->r = *in;
            out->i = 0;
        }
    }

    Bitmap_Image::Image ComplexData::get() const
    {
        if( empty() )
            return Bitmap_Image::Image();

        Bitmap_Image::Image image( _width, _height, 1u, 1u );
        uint8_t * out = image.data();

        const uint32_t size = _width * _height;
        const uint32_t middleX = _width  / 2;
        const uint32_t middleY = _height / 2;

        for( uint32_t inY = 0; inY < _height; ++inY ) {
            const uint32_t outY = (inY < middleY) ? middleY + inY : inY - middleY;

            for( uint32_t inX = 0; inX < _width; ++inX ) {
                const uint32_t outX = (inX < middleX) ? middleX + inX : inX - middleX;
                out[outY * _width + outX] = static_cast<uint8_t>(_data[inY * _width + inX].r / size + 0.5);
            }
        }

        return image;
    }

    void ComplexData::resize( uint32_t width_, uint32_t height_ )
    {
        if( ( width_ != _width || height_ != _height ) && width_ != 0 && height_ != 0 ) {
            _clean();

            const uint32_t size = width_ * height_;

            _data = reinterpret_cast<kiss_fft_cpx *>( malloc( size * sizeof(kiss_fft_cpx) ) );

            _width  = width_;
            _height = height_;
        }
    }

    kiss_fft_cpx * ComplexData::data()
    {
        return _data;
    }

    const kiss_fft_cpx * ComplexData::data() const
    {
        return _data;
    }

    uint32_t ComplexData::width() const
    {
        return _width;
    }

    uint32_t ComplexData::height() const
    {
        return _height;
    }

    bool ComplexData::empty() const
    {
        return _data == NULL;
    }

    void ComplexData::_clean()
    {
        if( _data != NULL ) {
            kiss_fft_free( _data );
            _data = NULL;
        }

        _width  = 0;
        _height = 0;
    }

    void ComplexData::_copy( const ComplexData & data )
    {
        _clean();

        resize( data._width, data._height );

        if( !empty() )
            memcpy( _data, data._data, _width * _height * sizeof(kiss_fft_cpx) );
    }

    void ComplexData::_swap( ComplexData & data )
    {
        std::swap( _data  , data._data );
        std::swap( _width , data._width );
        std::swap( _height, data._height );
    }

    FFTExecutor::FFTExecutor()
        : _planDirect  ( 0 )
        , _planInverse ( 0 )
        , _width       ( 0 )
        , _height      ( 0 )
    {
    }

    FFTExecutor::FFTExecutor( uint32_t width_, uint32_t height_ )
        : _planDirect  ( 0 )
        , _planInverse ( 0 )
        , _width       ( 0 )
        , _height      ( 0 )
    {
        initialize( width_, height_ );
    }

    FFTExecutor::~FFTExecutor()
    {
        _clean();
    }

    void FFTExecutor::initialize( uint32_t width_, uint32_t height_ )
    {
        if( width_ == 0 || height_ == 0 )
            throw imageException( "Invalid parameters for FFTExecutor" );

        _clean();

        const int dims[2] = { static_cast<int>(width_), static_cast<int>(height_) };
        _planDirect  = kiss_fftnd_alloc(dims, 2, false, 0, 0);
        _planInverse = kiss_fftnd_alloc(dims, 2, true , 0, 0);

        _width  = width_;
        _height = height_;
    }

    uint32_t FFTExecutor::width() const
    {
        return _width;
    }

    uint32_t FFTExecutor::height() const
    {
        return _height;
    }

    void FFTExecutor::directTransform( ComplexData & data )
    {
        directTransform( data, data );
    }

    void FFTExecutor::directTransform( const ComplexData & in, ComplexData & out )
    {
        if( _planDirect == 0 || _width != in.width() || _height != in.height() || _width != out.width() || _height != out.height() )
            throw imageException( "Invalid parameters for FFTExecutor" );

        kiss_fftnd( _planDirect, in.data(), out.data() );
    }

    void FFTExecutor::inverseTransform( ComplexData & data )
    {
        inverseTransform( data, data );
    }

    void FFTExecutor::inverseTransform( const ComplexData & in, ComplexData & out )
    {
        if( _planInverse == 0 || _width != in.width() || _height != in.height() || _width != out.width() || _height != out.height() )
            throw imageException( "Invalid parameters for FFTExecutor" );

        kiss_fftnd( _planInverse, in.data(), out.data() );
    }

    void FFTExecutor::complexMultiplication( const ComplexData & in1, const ComplexData & in2, ComplexData & out ) const
    {
        if( in1.width() != in2.width() || in1.height() != in2.height() || in1.width() != out.width() || in1.height() != out.height() ||
            in1.width() == 0 || in1.height() == 0 )
            throw imageException( "Invalid parameters for FFTExecutor" );

        // in1 = A + iB
        // in2 = C + iD
        // out = in1 * (-in2) = (A + iB) * (-C - iD) = - A * C - i(B * C) - i(A * D) + B * D

        const uint32_t size = in1.width() * in1.height();

        const kiss_fft_cpx * in1X = in1.data();
        const kiss_fft_cpx * in2X = in2.data();
        kiss_fft_cpx * outX = out.data();
        const kiss_fft_cpx * outXEnd = outX + size;

        for( ; outX != outXEnd; ++in1X, ++in2X, ++outX ) {
            outX->r = in1X->r * in2X->r - in1X->i * in2X->i;
            outX->i = in1X->r * in2X->i + in1X->i * in2X->r;
        }
    }

    void FFTExecutor::_clean()
    {
        if( _planDirect != 0 ) {
            kiss_fft_free( _planDirect );

            _planDirect = 0;
        }

        if( _planInverse != 0 ) {
            kiss_fft_free( _planInverse );

            _planInverse = 0;
        }

        _width  = 0;
        _height = 0;
    }
}
