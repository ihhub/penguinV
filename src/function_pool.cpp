/***************************************************************************
 *   penguinV: https://github.com/ihhub/penguinV                           *
 *   Copyright (C) 2017 - 2022                                             *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

#include "function_pool.h"
#include "function_pool_task.h"
#include "image_function_helper.h"
#include "parameter_validation.h"
#include "penguinv/penguinv.h"

namespace Function_Pool
{
    // This structure holds input parameters for some specific functions
    struct InputInfo
    {
        InputInfo()
            : minThreshold( 0 )
            , maxThreshold( 255 )
            , horizontalProjection( false )
            , coefficientA( 1 )
            , coefficientGamma( 1 )
            , extractChannelId( 255 )
            , horizontalFlip( false )
            , verticalFlip( false )
        {}

        uint8_t minThreshold; // for Threshold() function same as threshold
        uint8_t maxThreshold; // for Threshold() function
        bool horizontalProjection; // for ProjectionProfile() function
        double coefficientA; // for GammaCorrection() function
        double coefficientGamma; // for GammaCorrection() function
        uint8_t extractChannelId; // for ExtractChannel() function
        std::vector<uint8_t> lookupTable; // for LookupTable() function
        bool horizontalFlip;
        bool verticalFlip;
    };
    // This structure holds output data for some specific functions
    struct OutputInfo
    {
        std::vector<std::vector<uint32_t>> histogram; // for Histogram() function
        std::vector<std::vector<uint32_t>> projection; // for ProjectionProfile() function
        std::vector<uint32_t> sum; // for Sum() function
        std::vector<uint8_t> equality; // for IsEqual() function

        void resize( size_t count )
        {
            histogram.resize( count );
            projection.resize( count );
            sum.resize( count );
            equality.resize( count );
        }

        void getHistogram( std::vector<uint32_t> & histogram_ )
        {
            _getArray( histogram, histogram_ );
        }

        void getProjection( std::vector<uint32_t> & projection_ )
        {
            if ( projection.empty() )
                throw penguinVException( "Projection array is empty" );

            if ( projection_.size() == projection.front().size() ) {
                _getArray( projection, projection_ );
            }
            else {
                size_t totalSize = 0u;
                for ( size_t i = 0; i < projection.size(); ++i )
                    totalSize += projection[i].size();

                if ( projection_.size() != totalSize )
                    throw penguinVException( "Projection array is invalid" );

                uint32_t * out = projection_.data();
                for ( size_t i = 0; i < projection.size(); ++i ) {
                    std::vector<uint32_t>::const_iterator in = projection[i].begin();
                    std::vector<uint32_t>::const_iterator end = projection[i].end();

                    for ( ; in != end; ++in, ++out )
                        *out = *in;
                }
                projection.clear(); // to guarantee that no one can use it second time
            }
        }

        uint32_t getSum()
        {
            if ( sum.empty() )
                throw penguinVException( "Output array is empty" );

            uint32_t total = 0;

            for ( std::vector<uint32_t>::const_iterator value = sum.begin(); value != sum.end(); ++value )
                total += *value;

            sum.clear(); // to guarantee that no one can use it second time

            return total;
        }

        bool isEqual()
        {
            if ( equality.empty() )
                throw penguinVException( "Output array is empty" );

            bool equal = true;

            for ( std::vector<uint8_t>::const_iterator value = equality.begin(); value != equality.end(); ++value ) {
                if ( !( *value ) ) {
                    equal = false;
                    break;
                }
            }

            equality.clear(); // to guarantee that no one can use it second time

            return equal;
        }

    private:
        void _getArray( std::vector<std::vector<uint32_t>> & input, std::vector<uint32_t> & output ) const
        {
            if ( input.empty() )
                throw penguinVException( "Output array is empty" );

            output = input.front();

            if ( std::any_of( input.begin(), input.end(), [&output]( std::vector<uint32_t> & v ) { return v.size() != output.size(); } ) )
                throw penguinVException( "Returned histograms are not the same size" );

            for ( size_t i = 1; i < input.size(); ++i ) {
                std::vector<uint32_t>::iterator out = output.begin();
                std::vector<uint32_t>::const_iterator in = input[i].begin();
                std::vector<uint32_t>::const_iterator end = input[i].end();

                for ( ; in != end; ++in, ++out )
                    *out += *in;
            }

            input.clear(); // to guarantee that no one can use it second time
        }
    };

    class FunctionTask : public FunctionPoolTask
    {
    public:
        FunctionTask()
            : functionId( _none )
        {}

        virtual ~FunctionTask() {}

        // this is a list of image functions
        void AbsoluteDifference( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2, Image & out,
                                 uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
        {
            _setup( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
            _process( _AbsoluteDifference );
        }

        void BitwiseAnd( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2, Image & out, uint32_t startXOut,
                         uint32_t startYOut, uint32_t width, uint32_t height )
        {
            _setup( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
            _process( _BitwiseAnd );
        }

        void BitwiseOr( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2, Image & out, uint32_t startXOut,
                        uint32_t startYOut, uint32_t width, uint32_t height )
        {
            _setup( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
            _process( _BitwiseOr );
        }

        void BitwiseXor( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2, Image & out, uint32_t startXOut,
                         uint32_t startYOut, uint32_t width, uint32_t height )
        {
            _setup( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
            _process( _BitwiseXor );
        }

        void ConvertToGrayScale( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width,
                                 uint32_t height )
        {
            _setup( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
            _process( _ConvertToGrayScale );
        }

        void ConvertToRgb( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
        {
            _setup( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
            _process( _ConvertToRgb );
        }

        void ExtractChannel( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height,
                             uint8_t channelId )
        {
            _setup( in, startXIn, startYIn, out, startXOut, startYOut, width, height );

            _dataIn.extractChannelId = channelId;

            _process( _ExtractChannel );
        }

        void Flip( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height,
                   bool horizontal, bool vertical )
        {
            _setup( in, startXIn, startYIn, out, startXOut, startYOut, width, height );

            _dataIn.horizontalFlip = horizontal;
            _dataIn.verticalFlip = vertical;

            if ( _infoOut1->_size() > 1 ) {
                if ( _infoOut1->startY[0] == _infoOut1->startY[1] ) {
                    if ( horizontal ) {
                        for ( size_t i = 0u; i < _infoOut1->_size(); ++i )
                            _infoOut1->startX[i] = 2 * startXOut + width - ( _infoOut1->startX[i] + _infoOut1->width[i] );
                    }
                }
                else {
                    if ( vertical ) {
                        for ( size_t i = 0u; i < _infoOut1->_size(); ++i )
                            _infoOut1->startY[i] = 2 * startYOut + height - ( _infoOut1->startY[i] + _infoOut1->height[i] );
                    }
                }
            }

            _process( _Flip );
        }

        void GammaCorrection( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width,
                              uint32_t height, double a, double gamma )
        {
            _setup( in, startXIn, startYIn, out, startXOut, startYOut, width, height );

            if ( a < 0 || gamma < 0 )
                throw penguinVException( "Bad input parameters in image function" );

            _dataIn.coefficientA = a;
            _dataIn.coefficientGamma = gamma;

            _process( _GammaCorrection );
        }

        void Histogram( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, std::vector<uint32_t> & histogram )
        {
            _setup( image, x, y, width, height );
            _dataOut.resize( _infoIn1->_size() );
            _process( _Histogram );
            _dataOut.getHistogram( histogram );
        }

        void Invert( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
        {
            _setup( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
            _process( _Invert );
        }

        bool IsEqual( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2, uint32_t width, uint32_t height )
        {
            _setup( in1, startX1, startY1, in2, startX2, startY2, width, height );
            _dataOut.resize( _infoIn1->_size() );
            _process( _IsEqual );
            return _dataOut.isEqual();
        }

        void LookupTable( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height,
                          const std::vector<uint8_t> & table )
        {
            _setup( in, startXIn, startYIn, out, startXOut, startYOut, width, height );

            _dataIn.lookupTable = table;

            _process( _LookupTable );
        }

        void Merge( const Image & in1, uint32_t startXIn1, uint32_t startYIn1, const Image & in2, uint32_t startXIn2, uint32_t startYIn2, const Image & in3,
                    uint32_t startXIn3, uint32_t startYIn3, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
        {
            _setup( in1, startXIn1, startYIn1, in2, startXIn2, startYIn2, in3, startXIn3, startYIn3, out, startXOut, startYOut, width, height );

            _process( _Merge );
        }

        void Maximum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2, Image & out, uint32_t startXOut,
                      uint32_t startYOut, uint32_t width, uint32_t height )
        {
            _setup( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
            _process( _Maximum );
        }

        void Minimum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2, Image & out, uint32_t startXOut,
                      uint32_t startYOut, uint32_t width, uint32_t height )
        {
            _setup( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
            _process( _Minimum );
        }

        void ProjectionProfile( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, bool horizontal, std::vector<uint32_t> & projection )
        {
            _setup( image, x, y, width, height );
            _dataOut.resize( _infoIn1->_size() );
            _dataIn.horizontalProjection = horizontal;
            _process( _ProjectionProfile );

            projection.resize( horizontal ? width : height );
            _dataOut.getProjection( projection );
        }

        void Resize( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t widthIn, uint32_t heightIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                     uint32_t widthOut, uint32_t heightOut )
        {
            _setup( in, startXIn, startYIn, widthIn, heightIn, out, startXOut, startYOut, widthOut, heightOut );
            _process( _Resize );
        }

        void RgbToBgr( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
        {
            _setup( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
            _process( _RgbToBgr );
        }

        void Subtract( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2, Image & out, uint32_t startXOut,
                       uint32_t startYOut, uint32_t width, uint32_t height )
        {
            _setup( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
            _process( _Subtract );
        }

        void Split( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out1, uint32_t startXOut1, uint32_t startYOut1, Image & out2, uint32_t startXOut2,
                    uint32_t startYOut2, Image & out3, uint32_t startXOut3, uint32_t startYOut3, uint32_t width, uint32_t height )
        {
            _setup( in, startXIn, startYIn, out1, startXOut1, startYOut1, out2, startXOut2, startYOut2, out3, startXOut3, startYOut3, width, height );
            _process( _Split );
        }

        uint32_t Sum( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height )
        {
            _setup( image, x, y, width, height );
            _dataOut.resize( _infoIn1->_size() );
            _process( _Sum );
            return _dataOut.getSum();
        }

        void Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height,
                        uint8_t threshold )
        {
            _setup( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
            _dataIn.minThreshold = threshold;
            _process( _Threshold );
        }

        void Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height,
                        uint8_t minThreshold, uint8_t maxThreshold )
        {
            _setup( in, startXIn, startYIn, out, startXOut, startYOut, width, height );

            if ( minThreshold > maxThreshold )
                throw penguinVException( "Minimum threshold value is bigger than maximum threshold value" );

            _dataIn.minThreshold = minThreshold;
            _dataIn.maxThreshold = maxThreshold;
            _process( _ThresholdDouble );
        }

        void Transpose( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
        {
            _setup( in, startXIn, startYIn, width, height, out, startXOut, startYOut, height, width, true );
            _process( _Transpose );
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
            _Flip,
            _GammaCorrection,
            _Histogram,
            _Invert,
            _IsEqual,
            _LookupTable,
            _Maximum,
            _Merge,
            _Minimum,
            _ProjectionProfile,
            _Resize,
            _RgbToBgr,
            _Subtract,
            _Split,
            _Sum,
            _Threshold,
            _ThresholdDouble,
            _Transpose
        };

        virtual void _task( size_t taskId )
        {
            switch ( functionId ) {
            case _none:
                throw penguinVException( "Image function task is not setup" );
            case _AbsoluteDifference:
                penguinV::AbsoluteDifference( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId], _infoIn2->image, _infoIn2->startX[taskId],
                                              _infoIn2->startY[taskId], _infoOut1->image, _infoOut1->startX[taskId], _infoOut1->startY[taskId], _infoIn1->width[taskId],
                                              _infoIn1->height[taskId] );
                break;
            case _BitwiseAnd:
                penguinV::BitwiseAnd( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId], _infoIn2->image, _infoIn2->startX[taskId],
                                      _infoIn2->startY[taskId], _infoOut1->image, _infoOut1->startX[taskId], _infoOut1->startY[taskId], _infoIn1->width[taskId],
                                      _infoIn1->height[taskId] );
                break;
            case _BitwiseOr:
                penguinV::BitwiseOr( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId], _infoIn2->image, _infoIn2->startX[taskId],
                                     _infoIn2->startY[taskId], _infoOut1->image, _infoOut1->startX[taskId], _infoOut1->startY[taskId], _infoIn1->width[taskId],
                                     _infoIn1->height[taskId] );
                break;
            case _BitwiseXor:
                penguinV::BitwiseXor( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId], _infoIn2->image, _infoIn2->startX[taskId],
                                      _infoIn2->startY[taskId], _infoOut1->image, _infoOut1->startX[taskId], _infoOut1->startY[taskId], _infoIn1->width[taskId],
                                      _infoIn1->height[taskId] );
                break;
            case _ConvertToGrayScale:
                penguinV::ConvertToGrayScale( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId], _infoOut1->image, _infoOut1->startX[taskId],
                                              _infoOut1->startY[taskId], _infoIn1->width[taskId], _infoIn1->height[taskId] );
                break;
            case _ConvertToRgb:
                penguinV::ConvertToRgb( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId], _infoOut1->image, _infoOut1->startX[taskId],
                                        _infoOut1->startY[taskId], _infoIn1->width[taskId], _infoIn1->height[taskId] );
                break;
            case _ExtractChannel:
                penguinV::ExtractChannel( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId], _infoOut1->image, _infoOut1->startX[taskId],
                                          _infoOut1->startY[taskId], _infoIn1->width[taskId], _infoIn1->height[taskId], _dataIn.extractChannelId );
                break;
            case _Flip:
                penguinV::Flip( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId], _infoOut1->image, _infoOut1->startX[taskId],
                                _infoOut1->startY[taskId], _infoIn1->width[taskId], _infoIn1->height[taskId], _dataIn.horizontalFlip, _dataIn.verticalFlip );
                break;
            case _GammaCorrection:
                penguinV::GammaCorrection( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId], _infoOut1->image, _infoOut1->startX[taskId],
                                           _infoOut1->startY[taskId], _infoIn1->width[taskId], _infoIn1->height[taskId], _dataIn.coefficientA, _dataIn.coefficientGamma );
                break;
            case _Histogram:
                penguinV::Histogram( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId], _infoIn1->width[taskId], _infoIn1->height[taskId],
                                     _dataOut.histogram[taskId] );
                break;
            case _Invert:
                penguinV::Invert( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId], _infoOut1->image, _infoOut1->startX[taskId],
                                  _infoOut1->startY[taskId], _infoIn1->width[taskId], _infoIn1->height[taskId] );
                break;
            case _IsEqual:
                _dataOut.equality[taskId] = penguinV::IsEqual( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId], _infoIn2->image,
                                                               _infoIn2->startX[taskId], _infoIn2->startY[taskId], _infoIn1->width[taskId], _infoIn1->height[taskId] );
                break;
            case _LookupTable:
                penguinV::LookupTable( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId], _infoOut1->image, _infoOut1->startX[taskId],
                                       _infoOut1->startY[taskId], _infoIn1->width[taskId], _infoIn1->height[taskId], _dataIn.lookupTable );
                break;
            case _Maximum:
                penguinV::Maximum( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId], _infoIn2->image, _infoIn2->startX[taskId],
                                   _infoIn2->startY[taskId], _infoOut1->image, _infoOut1->startX[taskId], _infoOut1->startY[taskId], _infoIn1->width[taskId],
                                   _infoIn1->height[taskId] );
                break;
            case _Merge:
                penguinV::Merge( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId], _infoIn2->image, _infoIn2->startX[taskId], _infoIn2->startY[taskId],
                                 _infoIn3->image, _infoIn3->startX[taskId], _infoIn3->startY[taskId], _infoOut1->image, _infoOut1->startX[taskId],
                                 _infoOut1->startY[taskId], _infoIn1->width[taskId], _infoIn1->height[taskId] );
                break;
            case _Minimum:
                penguinV::Minimum( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId], _infoIn2->image, _infoIn2->startX[taskId],
                                   _infoIn2->startY[taskId], _infoOut1->image, _infoOut1->startX[taskId], _infoOut1->startY[taskId], _infoIn1->width[taskId],
                                   _infoIn1->height[taskId] );
                break;
            case _ProjectionProfile:
                penguinV::ProjectionProfile( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId], _infoIn1->width[taskId], _infoIn1->height[taskId],
                                             _dataIn.horizontalProjection, _dataOut.projection[taskId] );
                break;
            case _Resize:
                penguinV::Resize( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId], _infoIn1->width[taskId], _infoIn1->height[taskId],
                                  _infoOut1->image, _infoOut1->startX[taskId], _infoOut1->startY[taskId], _infoOut1->width[taskId], _infoOut1->height[taskId] );
                break;
            case _RgbToBgr:
                penguinV::RgbToBgr( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId], _infoOut1->image, _infoOut1->startX[taskId],
                                    _infoOut1->startY[taskId], _infoIn1->width[taskId], _infoIn1->height[taskId] );
                break;
            case _Subtract:
                penguinV::Subtract( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId], _infoIn2->image, _infoIn2->startX[taskId],
                                    _infoIn2->startY[taskId], _infoOut1->image, _infoOut1->startX[taskId], _infoOut1->startY[taskId], _infoIn1->width[taskId],
                                    _infoIn1->height[taskId] );
                break;
            case _Split:
                penguinV::Split( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId], _infoOut1->image, _infoOut1->startX[taskId],
                                 _infoOut1->startY[taskId], _infoOut2->image, _infoOut2->startX[taskId], _infoOut2->startY[taskId], _infoOut3->image,
                                 _infoOut3->startX[taskId], _infoOut3->startY[taskId], _infoIn1->width[taskId], _infoIn1->height[taskId] );
                break;
            case _Sum:
                _dataOut.sum[taskId]
                    = penguinV::Sum( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId], _infoIn1->width[taskId], _infoIn1->height[taskId] );
                break;
            case _Threshold:
                penguinV::Threshold( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId], _infoOut1->image, _infoOut1->startX[taskId],
                                     _infoOut1->startY[taskId], _infoIn1->width[taskId], _infoIn1->height[taskId], _dataIn.minThreshold );
                break;
            case _ThresholdDouble:
                penguinV::Threshold( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId], _infoOut1->image, _infoOut1->startX[taskId],
                                     _infoOut1->startY[taskId], _infoIn1->width[taskId], _infoIn1->height[taskId], _dataIn.minThreshold, _dataIn.maxThreshold );
                break;
            case _Transpose:
                penguinV::Transpose( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId], _infoOut1->image, _infoOut1->startX[taskId],
                                     _infoOut1->startY[taskId], _infoIn1->width[taskId], _infoIn1->height[taskId] );
                break;
            default:
                throw penguinVException( "Unknown image function task" );
            }
        }

    private:
        TaskName functionId;

        InputInfo _dataIn; // structure which holds some unique input parameters
        OutputInfo _dataOut; // structure which holds some unique output values

        void _process( TaskName id )
        {
            functionId = id;

            _processTask();
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

    Image AbsoluteDifference( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2, uint32_t width,
                              uint32_t height )
    {
        return Image_Function_Helper::AbsoluteDifference( AbsoluteDifference, in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    void AbsoluteDifference( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2, Image & out,
                             uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
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

    Image BitwiseAnd( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2, uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::BitwiseAnd( BitwiseAnd, in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    void BitwiseAnd( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2, Image & out, uint32_t startXOut,
                     uint32_t startYOut, uint32_t width, uint32_t height )
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

    Image BitwiseOr( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2, uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::BitwiseOr( BitwiseOr, in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    void BitwiseOr( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2, Image & out, uint32_t startXOut,
                    uint32_t startYOut, uint32_t width, uint32_t height )
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

    Image BitwiseXor( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2, uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::BitwiseXor( BitwiseXor, in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    void BitwiseXor( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2, Image & out, uint32_t startXOut,
                     uint32_t startYOut, uint32_t width, uint32_t height )
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

    void ConvertToGrayScale( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width,
                             uint32_t height )
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

    void ConvertToRgb( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
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

    void ExtractChannel( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height,
                         uint8_t channelId )
    {
        Image_Function::ValidateImageParameters( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
        Image_Function::VerifyGrayScaleImage( out );

        FunctionTask().ExtractChannel( in, startXIn, startYIn, out, startXOut, startYOut, width, height, channelId );
    }

    Image Flip( const Image & in, bool horizontal, bool vertical )
    {
        return Image_Function_Helper::Flip( Flip, in, horizontal, vertical );
    }

    void Flip( const Image & in, Image & out, bool horizontal, bool vertical )
    {
        Image_Function_Helper::Flip( Flip, in, out, horizontal, vertical );
    }

    Image Flip( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, bool horizontal, bool vertical )
    {
        return Image_Function_Helper::Flip( Flip, in, startXIn, startYIn, width, height, horizontal, vertical );
    }

    void Flip( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height,
               bool horizontal, bool vertical )
    {
        if ( !horizontal && !vertical ) {
            penguinV::Copy( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
            return;
        }

        Image_Function::ValidateImageParameters( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
        Image_Function::VerifyGrayScaleImage( in, out );

        FunctionTask().Flip( in, startXIn, startYIn, out, startXOut, startYOut, width, height, horizontal, vertical );
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

    void GammaCorrection( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height,
                          double a, double gamma )
    {
        FunctionTask().GammaCorrection( in, startXIn, startYIn, out, startXOut, startYOut, width, height, a, gamma );
    }

    std::vector<uint32_t> Histogram( const Image & image )
    {
        return Image_Function_Helper::Histogram( Histogram, image );
    }

    void Histogram( const Image & image, std::vector<uint32_t> & histogram )
    {
        Image_Function_Helper::Histogram( Histogram, image, histogram );
    }

    std::vector<uint32_t> Histogram( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::Histogram( Histogram, image, x, y, width, height );
    }

    void Histogram( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, std::vector<uint32_t> & histogram )
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

    void Invert( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        FunctionTask().Invert( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
    }

    bool IsEqual( const Image & in1, const Image & in2 )
    {
        Image_Function::ValidateImageParameters( in1, in2 );

        return Function_Pool::IsEqual( in1, 0, 0, in2, 0, 0, in1.width(), in1.height() );
    }

    bool IsEqual( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2, uint32_t width, uint32_t height )
    {
        return FunctionTask().IsEqual( in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    Image LookupTable( const Image & in, const std::vector<uint8_t> & table )
    {
        return Image_Function_Helper::LookupTable( LookupTable, in, table );
    }

    void LookupTable( const Image & in, Image & out, const std::vector<uint8_t> & table )
    {
        Image_Function_Helper::LookupTable( LookupTable, in, out, table );
    }

    Image LookupTable( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, const std::vector<uint8_t> & table )
    {
        return Image_Function_Helper::LookupTable( LookupTable, in, startXIn, startYIn, width, height, table );
    }

    void LookupTable( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height,
                      const std::vector<uint8_t> & table )
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

    Image Maximum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2, uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::Maximum( Maximum, in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    void Maximum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2, Image & out, uint32_t startXOut,
                  uint32_t startYOut, uint32_t width, uint32_t height )
    {
        FunctionTask().Maximum( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    Image Merge( const Image & in1, const Image & in2, const Image & in3 )
    {
        return Image_Function_Helper::Merge( Merge, in1, in2, in3 );
    }

    void Merge( const Image & in1, const Image & in2, const Image & in3, Image & out )
    {
        Image_Function_Helper::Merge( Merge, in1, in2, in3, out );
    }

    Image Merge( const Image & in1, uint32_t startXIn1, uint32_t startYIn1, const Image & in2, uint32_t startXIn2, uint32_t startYIn2, const Image & in3,
                 uint32_t startXIn3, uint32_t startYIn3, uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::Merge( Merge, in1, startXIn1, startYIn1, in2, startXIn2, startYIn2, in3, startXIn3, startYIn3, width, height );
    }

    void Merge( const Image & in1, uint32_t startXIn1, uint32_t startYIn1, const Image & in2, uint32_t startXIn2, uint32_t startYIn2, const Image & in3,
                uint32_t startXIn3, uint32_t startYIn3, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        FunctionTask().Merge( in1, startXIn1, startYIn1, in2, startXIn2, startYIn2, in3, startXIn3, startYIn3, out, startXOut, startYOut, width, height );
    }

    Image Minimum( const Image & in1, const Image & in2 )
    {
        return Image_Function_Helper::Minimum( Minimum, in1, in2 );
    }

    void Minimum( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function_Helper::Minimum( Minimum, in1, in2, out );
    }

    Image Minimum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2, uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::Minimum( Minimum, in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    void Minimum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2, Image & out, uint32_t startXOut,
                  uint32_t startYOut, uint32_t width, uint32_t height )
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

    void Normalize( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        const std::vector<uint32_t> histogram = Function_Pool::Histogram( in, startXIn, startYIn, width, height );
        if ( histogram.size() != 256u )
            throw penguinVException( "Histogram size is not equal to 256" );

        uint16_t minimum = 255u;
        uint16_t maximum = 0u;

        for ( uint16_t i = 0u; i < 256u; ++i ) {
            if ( histogram[i] > 0u ) {
                if ( maximum < i )
                    maximum = i;
                if ( minimum > i )
                    minimum = i;
            }
        }

        if ( minimum >= maximum ) {
            penguinV::Copy( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
        }
        else {
            const double correction = 255.0 / ( maximum - minimum );

            // We precalculate all values and store them in lookup table
            std::vector<uint8_t> value( 256 );

            for ( uint16_t i = 0; i < 256; ++i )
                value[i] = static_cast<uint8_t>( ( i - minimum ) * correction + 0.5 );

            FunctionTask().LookupTable( in, startXIn, startYIn, out, startXOut, startYOut, width, height, value );
        }
    }

    std::vector<uint32_t> ProjectionProfile( const Image & image, bool horizontal )
    {
        return Image_Function_Helper::ProjectionProfile( ProjectionProfile, image, horizontal );
    }

    void ProjectionProfile( const Image & image, bool horizontal, std::vector<uint32_t> & projection )
    {
        Function_Pool::ProjectionProfile( image, 0, 0, image.width(), image.height(), horizontal, projection );
    }

    std::vector<uint32_t> ProjectionProfile( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, bool horizontal )
    {
        return Image_Function_Helper::ProjectionProfile( ProjectionProfile, image, x, y, width, height, horizontal );
    }

    void ProjectionProfile( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, bool horizontal, std::vector<uint32_t> & projection )
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

    Image Resize( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t widthIn, uint32_t heightIn, uint32_t widthOut, uint32_t heightOut )
    {
        return Image_Function_Helper::Resize( Resize, in, startXIn, startYIn, widthIn, heightIn, widthOut, heightOut );
    }

    void Resize( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t widthIn, uint32_t heightIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                 uint32_t widthOut, uint32_t heightOut )
    {
        Image_Function::ValidateImageParameters( in, startXIn, startYIn, widthIn, heightIn );
        Image_Function::ValidateImageParameters( out, startXOut, startYOut, widthOut, heightOut );

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

    void RgbToBgr( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
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

    void Split( const Image & in, Image & out1, Image & out2, Image & out3 )
    {
        Image_Function::ValidateImageParameters( in, out1, out2 );
        Image_Function::ValidateImageParameters( in, out3 );
        Image_Function::VerifyRGBImage( in );
        Image_Function::VerifyGrayScaleImage( out1, out2, out3 );

        Image_Function_Helper::Split( Split, in, out1, out2, out3 );
    }

    void Split( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out1, uint32_t startXOut1, uint32_t startYOut1, Image & out2, uint32_t startXOut2,
                uint32_t startYOut2, Image & out3, uint32_t startXOut3, uint32_t startYOut3, uint32_t width, uint32_t height )
    {
        Image_Function::ValidateImageParameters( in, startXIn, startYIn, width, height );
        Image_Function::ValidateImageParameters( out1, startXOut1, startYOut1, out2, startXOut2, startYOut2, out3, startXOut3, startYOut3, width, height );
        Image_Function::VerifyRGBImage( in );
        Image_Function::VerifyGrayScaleImage( out1, out2, out3 );

        FunctionTask().Split( in, startXIn, startYIn, out1, startXOut1, startYOut1, out2, startXOut2, startYOut2, out3, startXOut3, startYOut3, width, height );
    }

    Image Subtract( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2, uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::Subtract( Subtract, in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    void Subtract( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2, Image & out, uint32_t startXOut,
                   uint32_t startYOut, uint32_t width, uint32_t height )
    {
        FunctionTask().Subtract( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    uint32_t Sum( const Image & image )
    {
        return Function_Pool::Sum( image, 0, 0, image.width(), image.height() );
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

    void Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height,
                    uint8_t threshold )
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

    Image Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, uint8_t minThreshold, uint8_t maxThreshold )
    {
        return Image_Function_Helper::Threshold( Threshold, in, startXIn, startYIn, width, height, minThreshold, maxThreshold );
    }

    void Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height,
                    uint8_t minThreshold, uint8_t maxThreshold )
    {
        FunctionTask().Threshold( in, startXIn, startYIn, out, startXOut, startYOut, width, height, minThreshold, maxThreshold );
    }

    Image Transpose( const Image & in )
    {
        return Image_Function_Helper::Transpose( Transpose, in );
    }

    void Transpose( const Image & in, Image & out )
    {
        Image_Function_Helper::Transpose( Transpose, in, out );
    }

    Image Transpose( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::Transpose( Transpose, in, startXIn, startYIn, width, height );
    }

    void Transpose( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        Image_Function::ValidateImageParameters( in, startXIn, startYIn, width, height );
        Image_Function::ValidateImageParameters( out, startXOut, startYOut, height, width );
        Image_Function::VerifyGrayScaleImage( in, out );

        FunctionTask().Transpose( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
    }
}
