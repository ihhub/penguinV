#include "function_pool_task.h"
#include "parameter_validation.h"

namespace
{
    uint32_t threadCount()
    {
        const uint32_t count = static_cast<uint32_t>(ThreadPoolMonoid::instance().threadCount());
        if( count == 0 )
            throw imageException( "Thread Pool is not initialized." );
        return count;
    }

    const uint32_t cacheSize = 16; // Remember: every CPU has it's own caching technique so processing time of
                                   // subsequent memory cells is much faster!
                                   // Change this value if you need to adjust to specific CPU. 16 bytes are set
                                   // for proper SSE/NEON support
}

namespace Function_Pool
{
    AreaInfo::AreaInfo( uint32_t x, uint32_t y, uint32_t width_, uint32_t height_, uint32_t count )
    {
        _calculate( x, y, width_, height_, count );
    }

    size_t AreaInfo::_size() const
    {
        return startX.size();
    }

    // this function makes a similar input data sorting like it is done in info parameter
    void AreaInfo::_copy( const AreaInfo & info, uint32_t x, uint32_t y, uint32_t width_, uint32_t height_ )
    {
        if( info._size() > 0 ) {
            bool yAxis = true;

            for( size_t i = 0; i < info._size() - 1; ++i ) {
                if( info.startX[i] != info.startX[i + 1] || info.width[i] != info.width[i + 1] ) {
                    yAxis = false;
                    break;
                }
            }

            _fill( x, y, width_, height_, static_cast<uint32_t>(info._size()), yAxis );
        }
    }

    // this function will sort out all input data into arrays for multithreading execution
    void AreaInfo::_calculate( uint32_t x, uint32_t y, uint32_t width_, uint32_t height_, uint32_t count )
    {
        uint32_t maximumXTaskCount = width_ / cacheSize;
        if( maximumXTaskCount == 0 )
            maximumXTaskCount = 1;
        if( maximumXTaskCount > count )
            maximumXTaskCount = count;

        uint32_t maximumYTaskCount = height_;
        if( maximumYTaskCount > count )
            maximumYTaskCount = count;

        count = (maximumYTaskCount >= maximumXTaskCount) ? maximumYTaskCount : maximumXTaskCount;

        _fill( x, y, width_, height_, count, maximumYTaskCount >= maximumXTaskCount );
    }

    // this function fills all arrays by necessary values
    void AreaInfo::_fill( uint32_t x, uint32_t y, uint32_t width_, uint32_t height_, uint32_t count, bool yAxis )
    {
        startX.resize( count );
        startY.resize( count );
        width.resize( count );
        height.resize( count );

        if( yAxis ) { // process by rows
            std::fill( startX.begin(), startX.end(), x );
            std::fill( width.begin(), width.end(), width_ );

            uint32_t remainValue = height_ % count;
            uint32_t previousValue = y;

            for( size_t i = 0; i < count; ++i ) {
                height[i] = height_ / count;
                if( remainValue > 0 ) {
                    --remainValue;
                    ++height[i];
                }
                startY[i] = previousValue;
                previousValue = startY[i] + height[i];
            }
        }
        else { // process by columns
            std::fill( startY.begin(), startY.end(), y );
            std::fill( height.begin(), height.end(), height_ );

            uint32_t remainValue = width_ % count;
            uint32_t previousValue = x;

            for( size_t i = 0; i < count; ++i ) {
                width[i] = width_ / count;
                if( remainValue > 0 ) {
                    --remainValue;
                    ++width[i];
                }
                startX[i] = previousValue;
                previousValue = startX[i] + width[i];
            }
        }
    }

    InputImageInfo::InputImageInfo( const Image & in, uint32_t x, uint32_t y, uint32_t width_, uint32_t height_, uint32_t count )
        : AreaInfo( x, y, width_, height_, count )
        , image( in )
    {
    }

    OutputImageInfo::OutputImageInfo( Image & in, uint32_t x, uint32_t y, uint32_t width_, uint32_t height_, uint32_t count )
        : AreaInfo( x, y, width_, height_, count )
        , image( in )
    {
    }

    FunctionPoolTask::FunctionPoolTask()
    {
    }

    FunctionPoolTask::~FunctionPoolTask()
    {
    }

    void FunctionPoolTask::_setup( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( image, x, y, width, height );

        if( !_ready() )
            throw imageException( "FunctionTask object was called multiple times!" );

        _infoIn1 = std::unique_ptr < InputImageInfo  >( new InputImageInfo ( image, x, y, width, height, threadCount() ) );
    }

    void FunctionPoolTask::_setup( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                 uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

        if( !_ready() )
            throw imageException( "FunctionTask object was called multiple times!" );

        _infoIn1 = std::unique_ptr < InputImageInfo  >( new InputImageInfo ( in1, startX1, startY1, width, height, threadCount() ) );
        _infoIn2 = std::unique_ptr < InputImageInfo  >( new InputImageInfo ( in2, startX2, startY2, width, height, threadCount() ) );
    }

    void FunctionPoolTask::_setup( const Image & in, uint32_t inX, uint32_t inY, Image & out, uint32_t outX, uint32_t outY, uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in, inX, inY, out, outX, outY, width, height );

        if( !_ready() )
            throw imageException( "FunctionTask object was called multiple times!" );

        _infoIn1 = std::unique_ptr < InputImageInfo  >( new InputImageInfo ( in, inX, inY, width, height, threadCount() ) );
        _infoOut = std::unique_ptr < OutputImageInfo >( new OutputImageInfo( out, outX, outY, width, height, threadCount() ) );
    }

    void FunctionPoolTask::_setup( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t widthIn, uint32_t heightIn,
                                   Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t widthOut, uint32_t heightOut )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, widthIn, heightIn );
        Image_Function::ParameterValidation( out, startXOut, startYOut, widthOut, heightOut );

        if( !_ready() )
            throw imageException( "FunctionTask object was called multiple times!" );

        _infoIn1 = std::unique_ptr < InputImageInfo  >( new InputImageInfo ( in, startXIn, startYIn, std::min( widthIn, widthOut ),
                                                                                                     std::min( heightIn, heightOut ), threadCount() ) );
        _infoOut = std::unique_ptr < OutputImageInfo >( new OutputImageInfo( out, startXOut, startYOut, widthOut, heightOut, threadCount() ) );

        _infoOut->_copy( *_infoIn1, startXOut, startYOut, widthOut, heightOut );
        _infoIn1->_copy( *_infoOut, startXIn, startYIn, widthIn, heightIn );
    }

    void FunctionPoolTask::_setup( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                                   Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );

        if( !_ready() )
            throw imageException( "FunctionTask object was called multiple times!" );

        _infoIn1 = std::unique_ptr < InputImageInfo  >( new InputImageInfo ( in1, startX1, startY1, width, height, threadCount() ) );
        _infoIn2 = std::unique_ptr < InputImageInfo  >( new InputImageInfo ( in2, startX2, startY2, width, height, threadCount() ) );
        _infoOut = std::unique_ptr < OutputImageInfo >( new OutputImageInfo( out, startXOut, startYOut, width, height, threadCount() ) );
    }

    void FunctionPoolTask::_processTask()
    {
        _run( _infoIn1->_size() );

        if( !_wait() ) {
            throw imageException( "An exception raised during task execution in function pool" );
        }
    }
}
