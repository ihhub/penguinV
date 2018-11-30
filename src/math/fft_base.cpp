#include "fft_base.h"
#include "../image_exception.h"

namespace FFT
{
    BaseFFTExecutor::BaseFFTExecutor() 
        : _width(0)
        , _height(0)
    {
    }

    void BaseFFTExecutor::initialize( uint32_t width_, uint32_t height_ )
    {
        if( width_ == 0 || height_ == 0 )
            throw imageException( "Invalid parameters for FFTExecutor::intialize()" );

        _clean();
        _width = width_;
        _height = height_;
        _makePlans();
    }

    uint32_t BaseFFTExecutor::width() const
    {
        return _width;
    }

    uint32_t BaseFFTExecutor::height() const
    {
        return _height;
    }

    void BaseFFTExecutor::_clean() 
    {
        _cleanPlans();
        _width = 0;
        _height = 0;
    }
}
