#include "fft.h"
#include "image_function.h"
#include "image_exception.h"

namespace FFT
{
    ComplexData::ComplexData()
    {
    }

    ComplexData::ComplexData( const PenguinV_Image::Image & image )
    {
        set( image );
    }

    ComplexData::ComplexData( const BaseComplexData<kiss_fft_cpx> & data )
    {
        _copy( data );
    }

    ComplexData::ComplexData( ComplexData && data )
    {
        _swap( data );
    }

    ComplexData::~ComplexData()
    {
        _clean();
    }

    void ComplexData::set( const PenguinV_Image::Image & image )
    {
        if( image.empty() || image.colorCount() != 1u )
            throw imageException( "Failed to allocate complex data for empty or coloured image" );

        _clean();

        const uint32_t size = image.width() * image.height();

        _allocateData(size * sizeof(kiss_fft_cpx));

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

    PenguinV_Image::Image ComplexData::get() const
    {
        if( empty() )
            return PenguinV_Image::Image();

        PenguinV_Image::Image image( _width, _height, 1u, 1u );
        uint8_t * out = image.data();

        const uint32_t size = _width * _height;
        const uint32_t middleX = _width  / 2;
        const uint32_t middleY = _height / 2;

        for( uint32_t inY = 0; inY < _height; ++inY ) {
            const uint32_t outY = (inY < middleY) ? middleY + inY : inY - middleY;

            for( uint32_t inX = 0; inX < _width; ++inX ) {
                const uint32_t outX = (inX < middleX) ? middleX + inX : inX - middleX;
                out[outY * _width + outX] = static_cast<uint8_t>(_data[inY * _width + inX].r / static_cast<float>(size) + 0.5);
            }
        }

        return image;
    }

    void ComplexData::_allocateData(size_t size) 
    {
         _data = reinterpret_cast<kiss_fft_cpx *>( malloc( size ) );
    }

    void ComplexData::_freeData()
    {
        kiss_fft_free( _data );
    }

    void ComplexData::_copyData( const BaseComplexData<kiss_fft_cpx> & data )
    {
        memcpy( _data, data.data(), _width * _height * sizeof(kiss_fft_cpx) );
    }

    FFTExecutor::FFTExecutor()
        : _planDirect  ( 0 )
        , _planInverse ( 0 )
    {
    }

    FFTExecutor::FFTExecutor( uint32_t width_, uint32_t height_ )
        : BaseFFTExecutor( width_, height_)
        , _planDirect    ( 0 )
        , _planInverse   ( 0 )
    {
        initialize( width_, height_ );
    }

    FFTExecutor::~FFTExecutor()
    {
        _clean();
    }

    bool FFTExecutor::dimensionsMatch( const ComplexData & data) const
    {
        return dimensionsMatch( data.width(), data.height() );
    }

    void FFTExecutor::directTransform( ComplexData & data )
    {
        directTransform( data, data );
    }

    void FFTExecutor::directTransform( const ComplexData & in, ComplexData & out )
    {
        if( _planDirect == 0 || !dimensionsMatch(in) || !dimensionsMatch(out) )
            throw imageException( "Invalid parameters for FFTExecutor::directTransform()" );

        kiss_fftnd( _planDirect, in.data(), out.data() );
    }

    void FFTExecutor::inverseTransform( ComplexData & data )
    {
        inverseTransform( data, data );
    }

    void FFTExecutor::inverseTransform( const ComplexData & in, ComplexData & out )
    {
        if( _planInverse == 0 || !dimensionsMatch(in) || !dimensionsMatch(out) )
            throw imageException( "Invalid parameters for FFTExecutor::inverseTransform()" );

        kiss_fftnd( _planInverse, in.data(), out.data() );
    }

    void FFTExecutor::complexMultiplication( const ComplexData & in1, const ComplexData & in2, ComplexData & out ) const
    {
        if( !in1.dimensionsMatch(in2) || !in1.dimensionsMatch(out) || in1.width() == 0 || in1.height() == 0)
            throw imageException( "Invalid parameters for FFTExecutor::complexMultiplication" );

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

    void FFTExecutor::_makePlans() 
    {
        const int dims[2] = { static_cast<int>(_width), static_cast<int>(_height) };
        _planDirect  = kiss_fftnd_alloc(dims, 2, false, 0, 0);
        _planInverse = kiss_fftnd_alloc(dims, 2, true , 0, 0);
    }

    void FFTExecutor::_cleanPlans()
    {
        if( _planDirect != 0 ) {
            kiss_fft_free( _planDirect );

            _planDirect = 0;
        }

        if( _planInverse != 0 ) {
            kiss_fft_free( _planInverse );

            _planInverse = 0;
        }
    }
}
