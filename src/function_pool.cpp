#include "image_function_helper.h"
#include "function_pool.h"
#include "parameter_validation.h"
#include "thread_pool.h"
#include "penguinv/penguinv.h"

namespace
{
    uint32_t threadCount()
    {
        uint32_t count = static_cast<uint32_t>(Thread_Pool::ThreadPoolMonoid::instance().threadCount());
        if( count == 0 )
            throw imageException( "Thread Pool is not initialized." );
        return count;
    }
}

namespace Function_Pool
{
    struct AreaInfo
    {
        AreaInfo( uint32_t x, uint32_t y, uint32_t width_, uint32_t height_, uint32_t count )
        {
            _calculate( x, y, width_, height_, count );
        }

        std::vector < uint32_t > startX; // start X position of image ROI
        std::vector < uint32_t > startY; // start Y position of image ROI
        std::vector < uint32_t > width;  // width of image ROI
        std::vector < uint32_t > height; // height of image ROI

        size_t _size() const
        {
            return startX.size();
        }

        // this function makes a similar input data sorting like it is done in info parameter
        void _copy( const AreaInfo & info, uint32_t x, uint32_t y, uint32_t width_, uint32_t height_ )
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
    private:
        static const uint32_t cacheSize = 16; // Remember: every CPU has it's own caching technique so processing time of
                                              // subsequent memory cells is much faster!
                                              // Change this value if you need to adjust to specific CPU. 16 bytes are set
                                              // for proper SSE/NEON support

        // this function will sort out all input data into arrays for multithreading execution
        void _calculate( uint32_t x, uint32_t y, uint32_t width_, uint32_t height_, uint32_t count )
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
        void _fill( uint32_t x, uint32_t y, uint32_t width_, uint32_t height_, uint32_t count, bool yAxis )
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
    };

    struct InputImageInfo : public AreaInfo
    {
        InputImageInfo( const Image & in, uint32_t x, uint32_t y, uint32_t width_, uint32_t height_, uint32_t count )
            : AreaInfo( x, y, width_, height_, count )
            , image( in )
        { }

        const Image & image;
    };

    struct OutputImageInfo : public AreaInfo
    {
        OutputImageInfo( Image & in, uint32_t x, uint32_t y, uint32_t width_, uint32_t height_, uint32_t count )
            : AreaInfo( x, y, width_, height_, count )
            , image( in )
        { }

        Image & image;
    };
    // This structure holds input parameters for some specific functions
    struct InputInfo
    {
        InputInfo()
            : minThreshold        ( 0 )
            , maxThreshold        ( 255 )
            , horizontalProjection( false )
            , coefficientA        ( 1 )
            , coefficientGamma    ( 1 )
            , extractChannelId    ( 255 )
        { }

        uint8_t minThreshold;      // for Threshold() function same as threshold
        uint8_t maxThreshold;      // for Threshold() function
        bool horizontalProjection; // for ProjectionProfile() function
        double coefficientA;       // for GammaCorrection() function
        double coefficientGamma;   // for GammaCorrection() function
        uint8_t extractChannelId;  // for ExtractChannel() function
        std::vector<uint8_t> lookupTable; // for LookupTable() function
    };
    // This structure holds output data for some specific functions
    struct OutputInfo
    {
        std::vector < std::vector < uint32_t > > histogram;  // for Histogram() function
        std::vector < std::vector < uint32_t > > projection; // for ProjectionProfile() function
        std::vector < uint32_t > sum;                        // for Sum() function
        std::vector < uint8_t > equality;                    // for IsEqual() function

        void resize( size_t count )
        {
            histogram.resize( count );
            projection.resize( count );
            sum.resize( count );
            equality.resize( count );
        }

        void getHistogram( std::vector <uint32_t> & histogram_ )
        {
            _getArray( histogram, histogram_ );
        }

        void getProjection( std::vector <uint32_t> & projection_ )
        {
            _getArray( projection, projection_ );
        }

        uint32_t getSum()
        {
            if( sum.empty() )
                throw imageException( "Output array is empty" );

            uint32_t total = 0;

            for( std::vector < uint32_t >::const_iterator value = sum.begin(); value != sum.end(); ++value )
                total += *value;

            sum.clear(); // to guarantee that no one can use it second time

            return total;
        }

        bool isEqual()
        {
            if( equality.empty() )
                throw imageException( "Output array is empty" );

            bool equal = true;

            for( std::vector < uint8_t >::const_iterator value = equality.begin(); value != equality.end(); ++value ) {
                if( !(*value) ) {
                    equal = false;
                    break;
                }
            }

            equality.clear(); // to guarantee that no one can use it second time

            return equal;
        }
    private:
        void _getArray( std::vector < std::vector < uint32_t > > & input, std::vector < uint32_t > & output ) const
        {
            if( input.empty() )
                throw imageException( "Output array is empty" );

            output = input.front();

            if( std::any_of( input.begin(), input.end(), [&output]( std::vector <uint32_t> & v ) { return v.size() != output.size(); } ) )
                throw imageException( "Returned histograms are not the same size" );

            for( size_t i = 1; i < input.size(); ++i ) {
                std::vector < uint32_t >::iterator       out = output.begin();
                std::vector < uint32_t >::const_iterator in  = input[i].begin();
                std::vector < uint32_t >::const_iterator end = input[i].end();

                for( ; in != end; ++in, ++out )
                    *out += *in;
            }

            input.clear(); // to guarantee that no one can use it second time
        }
    };

    class FunctionTask : public Thread_Pool::TaskProviderSingleton
    {
    public:
        FunctionTask()
            : functionId( _none )
        {}

        virtual ~FunctionTask() {}

        // this is a list of image functions
        void AbsoluteDifference( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                                 Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
        {
            _setup( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
            _process( _AbsoluteDifference );
        }

        void BitwiseAnd( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                         Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
        {
            _setup( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
            _process( _BitwiseAnd );
        }

        void BitwiseOr( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                        Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
        {
            _setup( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
            _process( _BitwiseOr );
        }

        void BitwiseXor( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                         Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
        {
            _setup( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
            _process( _BitwiseXor );
        }

        void ConvertToGrayScale( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                                 uint32_t width, uint32_t height )
        {
            _setup( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
            _process( _ConvertToGrayScale );
        }

        void ConvertToRgb( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                           uint32_t width, uint32_t height )
        {
            _setup( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
            _process( _ConvertToRgb );
        }

        void  ExtractChannel( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut,
                              uint32_t startYOut, uint32_t width, uint32_t height, uint8_t channelId )
        {
            _setup( in, startXIn, startYIn, out, startXOut, startYOut, width, height );

            _dataIn.extractChannelId = channelId;

            _process( _ExtractChannel );
        }

        void GammaCorrection( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                              uint32_t width, uint32_t height, double a, double gamma )
        {
            _setup( in, startXIn, startYIn, out, startXOut, startYOut, width, height );

            if( a < 0 || gamma < 0 )
                throw imageException( "Bad input parameters in image function" );

            _dataIn.coefficientA     = a;
            _dataIn.coefficientGamma = gamma;

            _process( _GammaCorrection );
        }

        void Histogram( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
                        std::vector < uint32_t > & histogram )
        {
            _setup( image, x, y, width, height );
            _dataOut.resize( _infoIn1->_size() );
            _process( _Histogram );
            _dataOut.getHistogram( histogram );
        }

        void Invert( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                     uint32_t width, uint32_t height )
        {
            _setup( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
            _process( _Invert );
        }

        bool IsEqual( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                      uint32_t width, uint32_t height )
        {
            _setup( in1, startX1, startY1, in2, startX2, startY2, width, height );
            _dataOut.resize( _infoIn1->_size() );
            _process( _IsEqual );
            return _dataOut.isEqual();
        }

        void LookupTable( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                          uint32_t width, uint32_t height, const std::vector < uint8_t > & table )
        {
            _setup( in, startXIn, startYIn, out, startXOut, startYOut, width, height );

            _dataIn.lookupTable = table;

            _process( _LookupTable );
        }

        void Maximum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                      Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
        {
            _setup( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
            _process( _Maximum );
        }

        void Minimum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                      Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
        {
            _setup( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
            _process( _Minimum );
        }

        void Normalize( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                        uint32_t width, uint32_t height )
        {
            _setup( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
            _process( _Normalize );
        }

        void ProjectionProfile( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, bool horizontal,
                                std::vector < uint32_t > & projection )
        {
            _setup( image, x, y, width, height );
            _dataOut.resize( _infoIn1->_size() );
            _dataIn.horizontalProjection = horizontal;
            _process( _ProjectionProfile );
            _dataOut.getProjection( projection );
        }

        void Resize( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t widthIn, uint32_t heightIn,
                     Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t widthOut, uint32_t heightOut )
        {
            _setup( in, startXIn, startYIn, widthIn, heightIn, out, startXOut, startYOut, widthOut, heightOut );
            _process( _Resize );
        }

        void RgbToBgr( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                       uint32_t width, uint32_t height )
        {
            _setup( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
            _process( _RgbToBgr );
        }

        void Subtract( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                       Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
        {
            _setup( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
            _process( _Subtract );
        }

        uint32_t Sum( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height )
        {
            _setup( image, x, y, width, height );
            _dataOut.resize( _infoIn1->_size() );
            _process( _Sum );
            return _dataOut.getSum();
        }

        void Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                        uint32_t width, uint32_t height, uint8_t threshold )
        {
            _setup( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
            _dataIn.minThreshold = threshold;
            _process( _Threshold );
        }

        void Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                        uint32_t width, uint32_t height, uint8_t minThreshold, uint8_t maxThreshold )
        {
            _setup( in, startXIn, startYIn, out, startXOut, startYOut, width, height );

            if( minThreshold > maxThreshold )
                throw imageException( "Minimum threshold value is bigger than maximum threshold value" );

            _dataIn.minThreshold = minThreshold;
            _dataIn.maxThreshold = maxThreshold;
            _process( _ThresholdDouble );
        }
    protected:
        enum TaskName // enumeration to define for thread which function need to execute
        {
            _none,
            _AbsoluteDifference,
            _BitwiseAnd,
            _BitwiseOr,
            _BitwiseXor,
            _ConvertToGrayScale,
            _ConvertToRgb,
            _ExtractChannel,
            _GammaCorrection,
            _Histogram,
            _Invert,
            _IsEqual,
            _LookupTable,
            _Maximum,
            _Minimum,
            _Normalize,
            _ProjectionProfile,
            _Resize,
            _RgbToBgr,
            _Subtract,
            _Sum,
            _Threshold,
            _ThresholdDouble
        };

        void _task( size_t taskId )
        {
            switch( functionId ) {
                case _none:
                    throw imageException( "Image function task is not setup" );
                case _AbsoluteDifference:
                    penguinV::AbsoluteDifference(
                        _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
                        _infoIn2->image, _infoIn2->startX[taskId], _infoIn2->startY[taskId],
                        _infoOut->image, _infoOut->startX[taskId], _infoOut->startY[taskId],
                        _infoIn1->width[taskId], _infoIn1->height[taskId] );
                    break;
                case _BitwiseAnd:
                    penguinV::BitwiseAnd( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
                                          _infoIn2->image, _infoIn2->startX[taskId], _infoIn2->startY[taskId],
                                          _infoOut->image, _infoOut->startX[taskId], _infoOut->startY[taskId],
                                          _infoIn1->width[taskId], _infoIn1->height[taskId] );
                    break;
                case _BitwiseOr:
                    penguinV::BitwiseOr( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
                                         _infoIn2->image, _infoIn2->startX[taskId], _infoIn2->startY[taskId],
                                         _infoOut->image, _infoOut->startX[taskId], _infoOut->startY[taskId],
                                         _infoIn1->width[taskId], _infoIn1->height[taskId] );
                    break;
                case _BitwiseXor:
                    penguinV::BitwiseXor( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
                                          _infoIn2->image, _infoIn2->startX[taskId], _infoIn2->startY[taskId],
                                          _infoOut->image, _infoOut->startX[taskId], _infoOut->startY[taskId],
                                          _infoIn1->width[taskId], _infoIn1->height[taskId] );
                    break;
                case _ConvertToGrayScale:
                    penguinV::ConvertToGrayScale( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
                                                  _infoOut->image, _infoOut->startX[taskId], _infoOut->startY[taskId],
                                                  _infoIn1->width[taskId], _infoIn1->height[taskId] );
                    break;
                case _ConvertToRgb:
                    penguinV::ConvertToRgb( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
                                            _infoOut->image, _infoOut->startX[taskId], _infoOut->startY[taskId],
                                            _infoIn1->width[taskId], _infoIn1->height[taskId] );
                    break;
                case _ExtractChannel:
                    penguinV::ExtractChannel(
                        _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
                        _infoOut->image, _infoOut->startX[taskId], _infoOut->startY[taskId],
                        _infoIn1->width[taskId], _infoIn1->height[taskId], _dataIn.extractChannelId );
                case _GammaCorrection:
                    penguinV::GammaCorrection(
                        _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
                        _infoOut->image, _infoOut->startX[taskId], _infoOut->startY[taskId],
                        _infoIn1->width[taskId], _infoIn1->height[taskId], _dataIn.coefficientA,
                        _dataIn.coefficientGamma );
                    break;
                case _Histogram:
                    penguinV::Histogram( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
                                         _infoIn1->width[taskId], _infoIn1->height[taskId], _dataOut.histogram[taskId] );
                    break;
                case _Invert:
                    penguinV::Invert( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
                                      _infoOut->image, _infoOut->startX[taskId], _infoOut->startY[taskId],
                                      _infoIn1->width[taskId], _infoIn1->height[taskId] );
                    break;
                case _IsEqual:
                    _dataOut.equality[taskId] = penguinV::IsEqual(
                        _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
                        _infoIn2->image, _infoIn2->startX[taskId], _infoIn2->startY[taskId],
                        _infoIn1->width[taskId], _infoIn1->height[taskId] );
                    break;
                case _LookupTable:
                    penguinV::LookupTable( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
                                           _infoOut->image, _infoOut->startX[taskId], _infoOut->startY[taskId],
                                           _infoIn1->width[taskId], _infoIn1->height[taskId], _dataIn.lookupTable );
                    break;
                case _Maximum:
                    penguinV::Maximum( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
                                       _infoIn2->image, _infoIn2->startX[taskId], _infoIn2->startY[taskId],
                                       _infoOut->image, _infoOut->startX[taskId], _infoOut->startY[taskId],
                                       _infoIn1->width[taskId], _infoIn1->height[taskId] );
                    break;
                case _Minimum:
                    penguinV::Minimum( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
                                       _infoIn2->image, _infoIn2->startX[taskId], _infoIn2->startY[taskId],
                                       _infoOut->image, _infoOut->startX[taskId], _infoOut->startY[taskId],
                                       _infoIn1->width[taskId], _infoIn1->height[taskId] );
                    break;
                case _Normalize:
                    penguinV::Normalize( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
                                         _infoOut->image, _infoOut->startX[taskId], _infoOut->startY[taskId],
                                         _infoIn1->width[taskId], _infoIn1->height[taskId] );
                    break;
                case _ProjectionProfile:
                    penguinV::ProjectionProfile(
                        _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
                        _infoIn1->width[taskId], _infoIn1->height[taskId],
                        _dataIn.horizontalProjection, _dataOut.projection[taskId] );
                    break;
                case _Resize:
                    penguinV::Resize( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
                                      _infoIn1->width[taskId], _infoIn1->height[taskId],
                                      _infoOut->image, _infoOut->startX[taskId], _infoOut->startY[taskId],
                                      _infoOut->width[taskId], _infoOut->height[taskId] );
                    break;
                case _RgbToBgr:
                    penguinV::RgbToBgr( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
                                        _infoOut->image, _infoOut->startX[taskId], _infoOut->startY[taskId],
                                        _infoIn1->width[taskId], _infoIn1->height[taskId] );
                    break;
                case _Subtract:
                    penguinV::Subtract( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
                                        _infoIn2->image, _infoIn2->startX[taskId], _infoIn2->startY[taskId],
                                        _infoOut->image, _infoOut->startX[taskId], _infoOut->startY[taskId],
                                        _infoIn1->width[taskId], _infoIn1->height[taskId] );
                    break;
                case _Sum:
                    _dataOut.sum[taskId] = penguinV::Sum(
                        _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
                        _infoIn1->width[taskId], _infoIn1->height[taskId] );
                    break;
                case _Threshold:
                    penguinV::Threshold( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
                                         _infoOut->image, _infoOut->startX[taskId], _infoOut->startY[taskId],
                                         _infoIn1->width[taskId], _infoIn1->height[taskId], _dataIn.minThreshold );
                    break;
                case _ThresholdDouble:
                    penguinV::Threshold( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
                                         _infoOut->image, _infoOut->startX[taskId], _infoOut->startY[taskId],
                                         _infoIn1->width[taskId], _infoIn1->height[taskId], _dataIn.minThreshold,
                                         _dataIn.maxThreshold );
                    break;
                default:
                    throw imageException( "Unknown image function task" );
            }
        }

    private:
        TaskName functionId;
        std::unique_ptr < InputImageInfo  > _infoIn1; // structure which holds information about first input image
        std::unique_ptr < InputImageInfo  > _infoIn2; // structure which holds information about second input image
        std::unique_ptr < OutputImageInfo > _infoOut; // structure which holds information about output image

        InputInfo  _dataIn;  // structure which holds some unique input parameters
        OutputInfo _dataOut; // structure which holds some unique output values

        // functions for setting up all parameters needed for multithreading and to validate input parameters
        void _setup( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height )
        {
            Image_Function::ParameterValidation( image, x, y, width, height );

            if( !_ready() )
                throw imageException( "FunctionTask object was called multiple times!" );

            _infoIn1 = std::unique_ptr < InputImageInfo  >( new InputImageInfo ( image, x, y, width, height, threadCount() ) );
        }

        void _setup( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                     uint32_t width, uint32_t height )
        {
            Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

            if( !_ready() )
                throw imageException( "FunctionTask object was called multiple times!" );

            _infoIn1 = std::unique_ptr < InputImageInfo  >( new InputImageInfo ( in1, startX1, startY1, width, height, threadCount() ) );
            _infoIn2 = std::unique_ptr < InputImageInfo  >( new InputImageInfo ( in2, startX2, startY2, width, height, threadCount() ) );
        }

        void _setup( const Image & in, uint32_t inX, uint32_t inY, Image & out, uint32_t outX, uint32_t outY, uint32_t width, uint32_t height )
        {
            Image_Function::ParameterValidation( in, inX, inY, out, outX, outY, width, height );

            if( !_ready() )
                throw imageException( "FunctionTask object was called multiple times!" );

            _infoIn1 = std::unique_ptr < InputImageInfo  >( new InputImageInfo ( in, inX, inY, width, height, threadCount() ) );
            _infoOut = std::unique_ptr < OutputImageInfo >( new OutputImageInfo( out, outX, outY, width, height, threadCount() ) );
        }

        void _setup( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t widthIn, uint32_t heightIn,
                     Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t widthOut, uint32_t heightOut )
        {
            Image_Function::ParameterValidation( in, startXIn, startYIn, widthIn, heightIn );
            Image_Function::ParameterValidation( out, startXOut, startYOut, widthOut, heightOut );

            if( !_ready() )
                throw imageException( "FunctionTask object was called multiple times!" );

            _infoIn1 = std::unique_ptr < InputImageInfo  >( new InputImageInfo ( in, startXIn, startYIn, std::min( widthIn, widthOut ), std::min( heightIn, heightOut ), threadCount() ) );
            _infoOut = std::unique_ptr < OutputImageInfo >( new OutputImageInfo( out, startXOut, startYOut, widthOut, heightOut, threadCount() ) );

            _infoOut->_copy( *_infoIn1, startXOut, startYOut, widthOut, heightOut );
            _infoIn1->_copy( *_infoOut, startXIn, startYIn, widthIn, heightIn );
        }

        void _setup( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                     Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
        {
            Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );

            if( !_ready() )
                throw imageException( "FunctionTask object was called multiple times!" );

            _infoIn1 = std::unique_ptr < InputImageInfo  >( new InputImageInfo ( in1, startX1, startY1, width, height, threadCount() ) );
            _infoIn2 = std::unique_ptr < InputImageInfo  >( new InputImageInfo ( in2, startX2, startY2, width, height, threadCount() ) );
            _infoOut = std::unique_ptr < OutputImageInfo >( new OutputImageInfo( out, startXOut, startYOut, width, height, threadCount() ) );
        }

        void _process( TaskName id ) // function which calls global thread pool and waits results from it
        {
            functionId = id;

            _run( _infoIn1->_size() );

            if( !_wait() ) {
                throw imageException( "An exception raised during task execution in function pool" );
            }
        }
    };

    // The list of global functions
    Image AbsoluteDifference( const Image & in1, const Image & in2 )
    {
        return Image_Function_Helper::AbsoluteDifference( AbsoluteDifference, in1, in2 );
    }

    void AbsoluteDifference( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function_Helper::AbsoluteDifference( AbsoluteDifference, in1, in2, out );
    }

    Image AbsoluteDifference( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                              uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::AbsoluteDifference( AbsoluteDifference, in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    void AbsoluteDifference( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                             Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        FunctionTask().AbsoluteDifference( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    Image BitwiseAnd( const Image & in1, const Image & in2 )
    {
        return Image_Function_Helper::BitwiseAnd( BitwiseAnd, in1, in2 );
    }

    void BitwiseAnd( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function_Helper::BitwiseAnd( BitwiseAnd, in1, in2, out );
    }

    Image BitwiseAnd( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                      uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::BitwiseAnd( BitwiseAnd, in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    void BitwiseAnd( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                     Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        FunctionTask().BitwiseAnd( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    Image BitwiseOr( const Image & in1, const Image & in2 )
    {
        return Image_Function_Helper::BitwiseOr( BitwiseOr, in1, in2 );
    }

    void BitwiseOr( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function_Helper::BitwiseOr( BitwiseOr, in1, in2, out );
    }

    Image BitwiseOr( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                     uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::BitwiseOr( BitwiseOr, in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    void BitwiseOr( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                    Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        FunctionTask().BitwiseOr( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    Image BitwiseXor( const Image & in1, const Image & in2 )
    {
        return Image_Function_Helper::BitwiseXor( BitwiseXor, in1, in2 );
    }

    void BitwiseXor( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function_Helper::BitwiseXor( BitwiseXor, in1, in2, out );
    }

    Image BitwiseXor( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                      uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::BitwiseXor( BitwiseXor, in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    void BitwiseXor( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                     Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        FunctionTask().BitwiseXor( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    Image ConvertToGrayScale( const Image & in )
    {
        return Image_Function_Helper::ConvertToGrayScale( ConvertToGrayScale, in );
    }

    void ConvertToGrayScale( const Image & in, Image & out )
    {
        Image_Function_Helper::ConvertToGrayScale( ConvertToGrayScale, in, out );
    }

    Image ConvertToGrayScale( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::ConvertToGrayScale( ConvertToGrayScale, in, startXIn, startYIn, width, height );
    }

    void ConvertToGrayScale( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                             uint32_t width, uint32_t height )
    {
        FunctionTask().ConvertToGrayScale( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
    }

    Image ConvertToRgb( const Image & in )
    {
        return Image_Function_Helper::ConvertToRgb( ConvertToRgb, in );
    }

    void ConvertToRgb( const Image & in, Image & out )
    {
        Image_Function_Helper::ConvertToRgb( ConvertToRgb, in, out );
    }

    Image ConvertToRgb( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::ConvertToRgb( ConvertToRgb, in, startXIn, startYIn, width, height );
    }

    void ConvertToRgb( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                       uint32_t width, uint32_t height )
    {
        FunctionTask().ConvertToRgb( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
    }

    Image ExtractChannel( const Image & in, uint8_t channelId )
    {
        return Image_Function_Helper::ExtractChannel( ExtractChannel, in, channelId );
    }

    void ExtractChannel( const Image & in, Image & out, uint8_t channelId )
    {
        Image_Function_Helper::ExtractChannel( ExtractChannel, in, out, channelId );
    }

    Image ExtractChannel( const Image & in, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint8_t channelId )
    {
        return Image_Function_Helper::ExtractChannel( ExtractChannel, in, x, y, width, height, channelId );
    }

    void ExtractChannel( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut,
                         uint32_t startYOut, uint32_t width, uint32_t height, uint8_t channelId )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
        Image_Function::VerifyGrayScaleImage( out );

        FunctionTask().ExtractChannel( in, startXIn, startYIn, out, startXOut, startYOut, width, height, channelId );
    }

    Image GammaCorrection( const Image & in, double a, double gamma )
    {
        return Image_Function_Helper::GammaCorrection( GammaCorrection, in, a, gamma );
    }

    void GammaCorrection( const Image & in, Image & out, double a, double gamma )
    {
        Image_Function_Helper::GammaCorrection( GammaCorrection, in, out, a, gamma );
    }

    Image GammaCorrection( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, double a, double gamma )
    {
        return Image_Function_Helper::GammaCorrection( GammaCorrection, in, startXIn, startYIn, width, height, a, gamma );
    }

    void GammaCorrection( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                          uint32_t width, uint32_t height, double a, double gamma )
    {
        FunctionTask().GammaCorrection( in, startXIn, startYIn, out, startXOut, startYOut, width, height, a, gamma );
    }

    std::vector < uint32_t > Histogram( const Image & image )
    {
        return Image_Function_Helper::Histogram( Histogram, image );
    }

    void Histogram( const Image & image, std::vector < uint32_t > & histogram )
    {
        Image_Function_Helper::Histogram( Histogram, image, histogram );
    }

    std::vector < uint32_t > Histogram( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::Histogram( Histogram, image, x, y, width, height );
    }

    void Histogram( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, std::vector < uint32_t > & histogram )
    {
        FunctionTask().Histogram( image, x, y, width, height, histogram );
    }

    Image Invert( const Image & in )
    {
        return Image_Function_Helper::Invert( Invert, in );
    }

    void Invert( const Image & in, Image & out )
    {
        Image_Function_Helper::Invert( Invert, in, out );
    }

    Image Invert( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::Invert( Invert, in, startXIn, startYIn, width, height );
    }

    void Invert( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                 uint32_t width, uint32_t height )
    {
        FunctionTask().Invert( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
    }

    bool IsEqual( const Image & in1, const Image & in2 )
    {
        Image_Function::ParameterValidation( in1, in2 );

        return IsEqual( in1, 0, 0, in2, 0, 0, in1.width(), in1.height() );
    }

    bool IsEqual( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                  uint32_t width, uint32_t height )
    {
        return FunctionTask().IsEqual( in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    Image LookupTable( const Image & in, const std::vector < uint8_t > & table )
    {
        return Image_Function_Helper::LookupTable( LookupTable, in, table );
    }

    void LookupTable( const Image & in, Image & out, const std::vector < uint8_t > & table )
    {
        Image_Function_Helper::LookupTable( LookupTable, in, out, table );
    }

    Image LookupTable( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height,
                       const std::vector < uint8_t > & table )
    {
        return Image_Function_Helper::LookupTable( LookupTable, in, startXIn, startYIn, width, height, table );
    }

    void LookupTable( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                      uint32_t width, uint32_t height, const std::vector < uint8_t > & table )
    {
        return FunctionTask().LookupTable( in, startXIn, startYIn, out, startXOut, startYOut, width, height, table );
    }

    Image Maximum( const Image & in1, const Image & in2 )
    {
        return Image_Function_Helper::Maximum( Maximum, in1, in2 );
    }

    void Maximum( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function_Helper::Maximum( Maximum, in1, in2, out );
    }

    Image Maximum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                   uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::Maximum( Maximum, in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    void Maximum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                  Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        FunctionTask().Maximum( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    Image Minimum( const Image & in1, const Image & in2 )
    {
        return Image_Function_Helper::Minimum( Minimum, in1, in2 );
    }

    void Minimum( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function_Helper::Minimum( Minimum, in1, in2, out );
    }

    Image Minimum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                   uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::Minimum( Minimum, in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    void Minimum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                  Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        FunctionTask().Minimum( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    Image Normalize( const Image & in )
    {
        return Image_Function_Helper::Normalize( Normalize, in );
    }

    void Normalize( const Image & in, Image & out )
    {
        Image_Function_Helper::Normalize( Normalize, in, out );
    }

    Image Normalize( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::Normalize( Normalize, in, startXIn, startYIn, width, height );
    }

    void Normalize( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                    uint32_t width, uint32_t height )
    {
        FunctionTask().Normalize( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
    }

    std::vector < uint32_t > ProjectionProfile( const Image & image, bool horizontal )
    {
        return Image_Function_Helper::ProjectionProfile( ProjectionProfile, image, horizontal );
    }

    void ProjectionProfile( const Image & image, bool horizontal, std::vector < uint32_t > & projection )
    {
        ProjectionProfile( image, 0, 0, image.width(), image.height(), horizontal, projection );
    }

    std::vector < uint32_t > ProjectionProfile( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, bool horizontal )
    {
        return Image_Function_Helper::ProjectionProfile( ProjectionProfile, image, x, y, width, height, horizontal );
    }

    void ProjectionProfile( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, bool horizontal,
                            std::vector < uint32_t > & projection )
    {
        FunctionTask().ProjectionProfile( image, x, y, width, height, horizontal, projection );
    }

    Image Resize( const Image & in, uint32_t widthOut, uint32_t heightOut )
    {
        return Image_Function_Helper::Resize( Resize, in, widthOut, heightOut );
    }

    void Resize( const Image & in, Image & out )
    {
        Image_Function_Helper::Resize( Resize, in, out );
    }

    Image Resize( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t widthIn, uint32_t heightIn,
                  uint32_t widthOut, uint32_t heightOut )
    {
        return Image_Function_Helper::Resize( Resize, in, startXIn, startYIn, widthIn, heightIn, widthOut, heightOut );
    }

    void Resize( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t widthIn, uint32_t heightIn,
                 Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t widthOut, uint32_t heightOut )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, widthIn, heightIn );
        Image_Function::ParameterValidation( out, startXOut, startYOut, widthOut, heightOut );

        FunctionTask().Resize( in, startXIn, startYIn, widthIn, heightIn, out, startXOut, startYOut, widthOut, heightOut );
    }

    Image RgbToBgr( const Image & in )
    {
        return Image_Function_Helper::RgbToBgr( RgbToBgr, in );
    }

    void RgbToBgr( const Image & in, Image & out )
    {
        Image_Function_Helper::RgbToBgr( RgbToBgr, in, out );
    }

    Image RgbToBgr( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::RgbToBgr( RgbToBgr, in, startXIn, startYIn, width, height );
    }

    void RgbToBgr( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                   uint32_t width, uint32_t height )
    {
        FunctionTask().RgbToBgr( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
    }

    Image Subtract( const Image & in1, const Image & in2 )
    {
        return Image_Function_Helper::Subtract( Subtract, in1, in2 );
    }

    void Subtract( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function_Helper::Subtract( Subtract, in1, in2, out );
    }

    Image Subtract( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                    uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::Subtract( Subtract, in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    void Subtract( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                   Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        FunctionTask().Subtract( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    uint32_t Sum( const Image & image )
    {
        return Sum( image, 0, 0, image.width(), image.height() );
    }

    uint32_t Sum( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height )
    {
        return FunctionTask().Sum( image, x, y, width, height );
    }

    Image Threshold( const Image & in, uint8_t threshold )
    {
        return Image_Function_Helper::Threshold( Threshold, in, threshold );
    }

    void Threshold( const Image & in, Image & out, uint8_t threshold )
    {
        Image_Function_Helper::Threshold( Threshold, in, out, threshold );
    }

    Image Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, uint8_t threshold )
    {
        return Image_Function_Helper::Threshold( Threshold, in, startXIn, startYIn, width, height, threshold );
    }

    void Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                    uint32_t width, uint32_t height, uint8_t threshold )
    {
        FunctionTask().Threshold( in, startXIn, startYIn, out, startXOut, startYOut, width, height, threshold );
    }

    Image Threshold( const Image & in, uint8_t minThreshold, uint8_t maxThreshold )
    {
        return Image_Function_Helper::Threshold( Threshold, in, minThreshold, maxThreshold );
    }

    void Threshold( const Image & in, Image & out, uint8_t minThreshold, uint8_t maxThreshold )
    {
        Image_Function_Helper::Threshold( Threshold, in, out, minThreshold, maxThreshold );
    }

    Image Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, uint8_t minThreshold,
                     uint8_t maxThreshold )
    {
        return Image_Function_Helper::Threshold( Threshold, in, startXIn, startYIn, width, height, minThreshold, maxThreshold );
    }

    void Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                    uint32_t width, uint32_t height, uint8_t minThreshold, uint8_t maxThreshold )
    {
        FunctionTask().Threshold( in, startXIn, startYIn, out, startXOut, startYOut, width, height, minThreshold, maxThreshold );
    }
}
