#include "function_pool.h"
#include "image_function.h"
#include "thread_pool.h"

namespace
{
	uint32_t threadCount()
	{
		uint32_t count = static_cast<uint32_t>(Thread_Pool::ThreadPoolMonoid::instance().threadCount());
		if (count == 0)
			throw imageException("Thread Pool is not initialized.");
		return count;
	};
};

namespace Function_Pool
{
	struct AreaInfo
	{
		AreaInfo( uint32_t x, uint32_t y, uint32_t width_, uint32_t height_, uint32_t count )
		{
			_calculate( x, y, width_, height_, count );
		};

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
		void _fill(uint32_t x, uint32_t y, uint32_t width_, uint32_t height_, uint32_t count, bool yAxis)
		{
			startX.resize( count );
			startY.resize( count );
			width .resize( count );
			height.resize( count );

			if( yAxis ) { // process by rows
				std::fill( startX.begin(), startX.end(), x );
				std::fill( width.begin() , width.end() , width_ );

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
		InputImageInfo(const Image & in, uint32_t x, uint32_t y, uint32_t width_, uint32_t height_, uint32_t count)
			: AreaInfo( x, y, width_, height_, count)
			, image(in)
		{ }

		const Image & image;
	};

	struct OutputImageInfo : public AreaInfo
	{
		OutputImageInfo(Image & in, uint32_t x, uint32_t y, uint32_t width_, uint32_t height_, uint32_t count)
			: AreaInfo( x, y, width_, height_, count)
			, image(in)
		{ }

		Image & image;
	};
	// This structure holds input parameters for some specific functions
	struct InputInfo
	{
		InputInfo()
			: minThreshold        (0)
			, maxThreshold        (255)
			, horizontalProjection(false)
			, coefficientA        (1)
			, coefficientGamma    (1)
		{ }

		uint8_t minThreshold;      // for Threshold() function same as threshold
		uint8_t maxThreshold;      // for Threshold() function
		bool horizontalProjection; // for ProjectionProfile() function
		double coefficientA;       // for GammaCorrection() function
		double coefficientGamma;   // for GammaCorrection() function
	};
	// This structure holds output data for some specific functions
	struct OutputInfo
	{
		std::vector < std::vector < uint32_t > > histogram;  // for Histogram() function
		std::vector < std::vector < uint32_t > > projection; // for ProjectionProfile() function
		std::vector < uint32_t > sum;						 // for Sum() function
		std::vector < uint8_t > equality;                    // for IsEqual() function

		void resize(size_t count)
		{
			histogram .resize(count);
			projection.resize(count);
			sum       .resize(count);
			equality  .resize(count);
		}

		void getHistogram(std::vector <uint32_t> & histogram_ )
		{
			_getArray( histogram, histogram_ );
		}

		void getProjection(std::vector <uint32_t> & projection_ )
		{
			_getArray( projection, projection_ );
		}

		uint32_t getSum()
		{
			if( sum.empty() )
				throw imageException("Output array is empty");

			uint32_t total = 0;

			for( std::vector < uint32_t >::const_iterator value = sum.begin(); value != sum.end(); ++value )
				total += *value;

			sum.clear(); // to guarantee that no one can use it second time

			return total;
		}

		bool isEqual()
		{
			if( equality.empty() )
				throw imageException("Output array is empty");

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
				throw imageException("Output array is empty");

			output = input.front();

			if( std::any_of( input.begin(), input.end(), [&output](std::vector <uint32_t> & v) { return v.size() != output.size(); } ) )
				throw imageException("Returned histograms are not the same size");

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
			: functionId(_none)
		{ };

		virtual ~FunctionTask() { };

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

		void GammaCorrection( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
							  uint32_t width, uint32_t height, double a, double gamma )
		{
			_setup( in, startXIn, startYIn, out, startXOut, startYOut, width, height );

			if( a < 0 || gamma < 0 )
				throw imageException("Bad input parameters in image function");

			_dataIn.coefficientA     = a;
			_dataIn.coefficientGamma = gamma;

			_process( _GammaCorrection );
		}

		void Histogram( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
						std::vector < uint32_t > & histogram )
		{
			_setup( image, x, y, width, height );
			_dataOut.resize(_infoIn1->_size());
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
			_dataOut.resize(_infoIn1->_size());
			_process( _IsEqual );
			return _dataOut.isEqual();
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
			_dataOut.resize(_infoIn1->_size());
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
			_dataOut.resize(_infoIn1->_size());
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
				throw imageException("Minimum threshold value is bigger than maximum threshold value");

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
			_GammaCorrection,
			_Histogram,
			_Invert,
			_IsEqual,
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

		void _task(size_t taskId)
		{
			switch(functionId) {
			case _none:
				throw imageException("Image function task is not setup");
			case _AbsoluteDifference:
				Image_Function::AbsoluteDifference(
											_infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
											_infoIn2->image, _infoIn2->startX[taskId], _infoIn2->startY[taskId],
											_infoOut->image, _infoOut->startX[taskId], _infoOut->startY[taskId],
											_infoIn1->width[taskId], _infoIn1->height[taskId] );
				break;
			case _BitwiseAnd:
				Image_Function::BitwiseAnd( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
											_infoIn2->image, _infoIn2->startX[taskId], _infoIn2->startY[taskId],
											_infoOut->image, _infoOut->startX[taskId], _infoOut->startY[taskId],
											_infoIn1->width[taskId], _infoIn1->height[taskId] );
				break;
			case _BitwiseOr:
				Image_Function::BitwiseOr(  _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
											_infoIn2->image, _infoIn2->startX[taskId], _infoIn2->startY[taskId],
											_infoOut->image, _infoOut->startX[taskId], _infoOut->startY[taskId],
											_infoIn1->width[taskId], _infoIn1->height[taskId] );
				break;
			case _BitwiseXor:
				Image_Function::BitwiseXor( _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
											_infoIn2->image, _infoIn2->startX[taskId], _infoIn2->startY[taskId],
											_infoOut->image, _infoOut->startX[taskId], _infoOut->startY[taskId],
											_infoIn1->width[taskId], _infoIn1->height[taskId] );
				break;
			case _ConvertToGrayScale:
				Image_Function::ConvertToGrayScale(_infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
											_infoOut->image, _infoOut->startX[taskId], _infoOut->startY[taskId],
											_infoIn1->width[taskId], _infoIn1->height[taskId] );
				break;
			case _ConvertToRgb:
				Image_Function::ConvertToRgb(_infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
											_infoOut->image, _infoOut->startX[taskId], _infoOut->startY[taskId],
											_infoIn1->width[taskId], _infoIn1->height[taskId] );
				break;
			case _GammaCorrection:
				Image_Function::GammaCorrection(
											_infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
											_infoOut->image, _infoOut->startX[taskId], _infoOut->startY[taskId],
											_infoIn1->width[taskId], _infoIn1->height[taskId], _dataIn.coefficientA,
											_dataIn.coefficientGamma);
				break;
			case _Histogram:
				Image_Function::Histogram(  _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
											_infoIn1->width[taskId], _infoIn1->height[taskId], _dataOut.histogram[taskId] );
				break;
			case _Invert:
				Image_Function::Invert(     _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
											_infoOut->image, _infoOut->startX[taskId], _infoOut->startY[taskId],
											_infoIn1->width[taskId], _infoIn1->height[taskId] );
				break;
			case _IsEqual:
				_dataOut.equality[taskId] = Image_Function::IsEqual(
											_infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
											_infoIn2->image, _infoIn2->startX[taskId], _infoIn2->startY[taskId],
											_infoIn1->width[taskId], _infoIn1->height[taskId] );
				break;
			case _Maximum:
				Image_Function::Maximum(    _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
											_infoIn2->image, _infoIn2->startX[taskId], _infoIn2->startY[taskId],
											_infoOut->image, _infoOut->startX[taskId], _infoOut->startY[taskId],
											_infoIn1->width[taskId], _infoIn1->height[taskId] );
				break;
			case _Minimum:
				Image_Function::Minimum(    _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
											_infoIn2->image, _infoIn2->startX[taskId], _infoIn2->startY[taskId],
											_infoOut->image, _infoOut->startX[taskId], _infoOut->startY[taskId],
											_infoIn1->width[taskId], _infoIn1->height[taskId] );
				break;
			case _Normalize:
				Image_Function::Normalize(  _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
											_infoOut->image, _infoOut->startX[taskId], _infoOut->startY[taskId],
											_infoIn1->width[taskId], _infoIn1->height[taskId] );
				break;
			case _ProjectionProfile:
				Image_Function::ProjectionProfile(
											_infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
											_infoIn1->width[taskId], _infoIn1->height[taskId],
											_dataIn.horizontalProjection, _dataOut.projection[taskId] );
				break;
			case _Resize:
				Image_Function::Resize(     _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
											_infoIn1->width[taskId], _infoIn1->height[taskId],
											_infoOut->image, _infoOut->startX[taskId], _infoOut->startY[taskId],
											_infoOut->width[taskId], _infoOut->height[taskId] );
				break;
			case _RgbToBgr:
				Image_Function::RgbToBgr(   _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
											_infoOut->image, _infoOut->startX[taskId], _infoOut->startY[taskId],
											_infoIn1->width[taskId], _infoIn1->height[taskId] );
				break;
			case _Subtract:
				Image_Function::Subtract(   _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
											_infoIn2->image, _infoIn2->startX[taskId], _infoIn2->startY[taskId],
											_infoOut->image, _infoOut->startX[taskId], _infoOut->startY[taskId],
											_infoIn1->width[taskId], _infoIn1->height[taskId] );
				break;
			case _Sum:
				_dataOut.sum[taskId] = Image_Function::Sum(
											_infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
											_infoIn1->width[taskId], _infoIn1->height[taskId] );
				break;
			case _Threshold:
				Image_Function::Threshold(  _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
											_infoOut->image, _infoOut->startX[taskId], _infoOut->startY[taskId],
											_infoIn1->width[taskId], _infoIn1->height[taskId], _dataIn.minThreshold );
				break;
			case _ThresholdDouble:
				Image_Function::Threshold(  _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
											_infoOut->image, _infoOut->startX[taskId], _infoOut->startY[taskId],
											_infoIn1->width[taskId], _infoIn1->height[taskId], _dataIn.minThreshold,
											_dataIn.maxThreshold);
				break;
			default:
				throw imageException("Unknown image function task");
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
				throw imageException("FunctionTask object was called multiple times!");

			_infoIn1 = std::unique_ptr < InputImageInfo  >( new InputImageInfo ( image, x , y, width, height, threadCount() ) );
		}

		void _setup( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					 uint32_t width, uint32_t height )
		{
			Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

			if( !_ready() )
				throw imageException("FunctionTask object was called multiple times!");

			_infoIn1 = std::unique_ptr < InputImageInfo  >( new InputImageInfo ( in1, startX1  , startY1  , width, height, threadCount() ) );
			_infoIn2 = std::unique_ptr < InputImageInfo  >( new InputImageInfo ( in2, startX2  , startY2  , width, height, threadCount() ) );
		}

		void _setup( const Image & in, uint32_t inX, uint32_t inY, Image & out, uint32_t outX, uint32_t outY, uint32_t width, uint32_t height )
		{
			Image_Function::ParameterValidation( in, inX, inY, out, outX, outY, width, height );

			if( !_ready() )
				throw imageException("FunctionTask object was called multiple times!");

			_infoIn1 = std::unique_ptr < InputImageInfo  >( new InputImageInfo ( in , inX , inY , width, height, threadCount() ) );
			_infoOut = std::unique_ptr < OutputImageInfo >( new OutputImageInfo( out, outX, outY, width, height, threadCount() ) );
		}

		void _setup( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t widthIn, uint32_t heightIn,
					 Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t widthOut, uint32_t heightOut )
		{
			Image_Function::ParameterValidation( in , startXIn , startYIn , widthIn , heightIn  );
			Image_Function::ParameterValidation( out, startXOut, startYOut, widthOut, heightOut );

			if( !_ready() )
				throw imageException("FunctionTask object was called multiple times!");

			_infoIn1 = std::unique_ptr < InputImageInfo  >( new InputImageInfo ( in , startXIn , startYIn , std::min(widthIn, widthOut), std::min(heightIn, heightOut), threadCount() ) );
			_infoOut = std::unique_ptr < OutputImageInfo >( new OutputImageInfo( out, startXOut, startYOut, widthOut, heightOut, threadCount() ) );

			_infoOut->_copy( *_infoIn1, startXOut, startYOut, widthOut, heightOut );
			_infoIn1->_copy( *_infoOut, startXIn , startYIn , widthIn , heightIn  );
		}

		void _setup( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					 Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
		{
			Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );

			if( !_ready() )
				throw imageException("FunctionTask object was called multiple times!");

			_infoIn1 = std::unique_ptr < InputImageInfo  >( new InputImageInfo ( in1, startX1  , startY1  , width, height, threadCount() ) );
			_infoIn2 = std::unique_ptr < InputImageInfo  >( new InputImageInfo ( in2, startX2  , startY2  , width, height, threadCount() ) );
			_infoOut = std::unique_ptr < OutputImageInfo >( new OutputImageInfo( out, startXOut, startYOut, width, height, threadCount() ) );
		}

		void _process(TaskName id) // function which calls global thread pool and waits results from it
		{
			functionId = id;

			_run( _infoIn1->_size() );

			if( !_wait() ) {
				throw imageException("An exception raised during task execution in function pool");
			}
		}
	};

	// The list of global functions
	Image AbsoluteDifference( const Image & in1, const Image & in2 )
	{
		Image_Function::ParameterValidation( in1, in2 );

		Image out( in1.width(), in1.height() );

		AbsoluteDifference( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void AbsoluteDifference( const Image & in1, const Image & in2, Image & out )
	{
		Image_Function::ParameterValidation( in1, in2, out );

		AbsoluteDifference( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
	}

	Image AbsoluteDifference( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
							  uint32_t width, uint32_t height )
	{
		Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

		Image out( width, height );

		AbsoluteDifference( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void AbsoluteDifference( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
							 Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
	{
		FunctionTask().AbsoluteDifference(in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height);
	}

	Image BitwiseAnd( const Image & in1, const Image & in2 )
	{
		Image_Function::ParameterValidation( in1, in2 );

		Image out( in1.width(), in1.height() );

		BitwiseAnd( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void BitwiseAnd( const Image & in1, const Image & in2, Image & out )
	{
		Image_Function::ParameterValidation( in1, in2, out );

		BitwiseAnd( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
	}

	Image BitwiseAnd( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					  uint32_t width, uint32_t height )
	{
		Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

		Image out( width, height );

		BitwiseAnd( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void BitwiseAnd( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					 Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
	{
		FunctionTask().BitwiseAnd(in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height);
	}

	Image BitwiseOr( const Image & in1, const Image & in2 )
	{
		Image_Function::ParameterValidation( in1, in2 );

		Image out( in1.width(), in1.height() );

		BitwiseOr( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void BitwiseOr( const Image & in1, const Image & in2, Image & out )
	{
		Image_Function::ParameterValidation( in1, in2, out );

		BitwiseOr( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
	}

	Image BitwiseOr( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					 uint32_t width, uint32_t height )
	{
		Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

		Image out( width, height );

		BitwiseOr( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void BitwiseOr( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
	{
		FunctionTask().BitwiseOr(in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height);
	}

	Image BitwiseXor( const Image & in1, const Image & in2 )
	{
		Image_Function::ParameterValidation( in1, in2 );

		Image out( in1.width(), in1.height() );

		BitwiseXor( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void BitwiseXor( const Image & in1, const Image & in2, Image & out )
	{
		Image_Function::ParameterValidation( in1, in2, out );

		BitwiseXor( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
	}

	Image BitwiseXor( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					  uint32_t width, uint32_t height )
	{
		Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

		Image out( width, height );

		BitwiseXor( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void BitwiseXor( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					 Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
	{
		FunctionTask().BitwiseXor(in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height);
	}

	Image ConvertToGrayScale( const Image & in )
	{
		Image_Function::ParameterValidation( in );

		Image out( in.width(), in.height() );

		ConvertToGrayScale(in, 0, 0, out, 0, 0, out.width(), out.height());

		return out;
	}

	void ConvertToGrayScale( const Image & in, Image & out )
	{
		Image_Function::ParameterValidation( in, out );

		ConvertToGrayScale(in, 0, 0, out, 0, 0, out.width(), out.height());
	}

	Image ConvertToGrayScale( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
	{
		Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

		Image out( width, height );

		ConvertToGrayScale( in, startXIn, startYIn, out, 0, 0, width, height );

		return out;
	}

	void ConvertToGrayScale( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
							 uint32_t width, uint32_t height )
	{
		FunctionTask().ConvertToGrayScale(in, startXIn, startYIn, out, startXOut, startYOut, width, height);
	}

	Image ConvertToRgb( const Image & in )
	{
		Image_Function::ParameterValidation( in );

		Image out( in.width(), in.height(), RGB );

		ConvertToRgb(in, 0, 0, out, 0, 0, out.width(), out.height());

		return out;
	}

	void ConvertToRgb( const Image & in, Image & out )
	{
		Image_Function::ParameterValidation( in, out );

		ConvertToRgb(in, 0, 0, out, 0, 0, out.width(), out.height());
	}

	Image ConvertToRgb( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
	{
		Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

		Image out( width, height, RGB );

		ConvertToRgb( in, startXIn, startYIn, out, 0, 0, width, height );

		return out;
	}

	void ConvertToRgb( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
					   uint32_t width, uint32_t height )
	{
		FunctionTask().ConvertToRgb(in, startXIn, startYIn, out, startXOut, startYOut, width, height);
	}

	Image GammaCorrection( const Image & in, double a, double gamma )
	{
		Image_Function::ParameterValidation( in );

		Image out( in.width(), in.height() );

		GammaCorrection( in, 0, 0, out, 0, 0, out.width(), out.height(), a, gamma );

		return out;
	}

	void GammaCorrection( const Image & in, Image & out, double a, double gamma )
	{
		Image_Function::ParameterValidation( in, out );

		GammaCorrection( in, 0, 0, out, 0, 0, out.width(), out.height(), a, gamma );
	}

	Image GammaCorrection( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, double a, double gamma )
	{
		Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

		Image out( width, height );

		GammaCorrection( in, startXIn, startYIn, out, 0, 0, width, height, a, gamma );

		return out;
	}

	void GammaCorrection( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
						  uint32_t width, uint32_t height, double a, double gamma )
	{
		FunctionTask().GammaCorrection( in, startXIn, startYIn, out, startXOut, startYOut, width, height, a, gamma );
	}

	std::vector < uint32_t > Histogram( const Image & image )
	{
		std::vector < uint32_t > histogram;

		Histogram( image, 0, 0, image.width(), image.height(), histogram );

		return histogram;
	}

	void Histogram( const Image & image, std::vector < uint32_t > & histogram )
	{
		Histogram( image, 0, 0, image.width(), image.height(), histogram );
	}

	std::vector < uint32_t > Histogram( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height )
	{
		Image_Function::ParameterValidation( image, x, y, width, height );

		std::vector < uint32_t > histogram;

		Histogram( image, x, y, width, height, histogram );

		return histogram;
	}

	void Histogram( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, std::vector < uint32_t > & histogram )
	{
		FunctionTask().Histogram(image, x, y, width, height, histogram );
	}

	Image Invert( const Image & in )
	{
		Image_Function::ParameterValidation( in );

		Image out( in.width(), in.height() );

		Invert( in, 0, 0, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void Invert( const Image & in, Image & out )
	{
		Image_Function::ParameterValidation( in, out );

		Invert( in, 0, 0, out, 0, 0, out.width(), out.height() );
	}

	Image Invert( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
	{
		Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

		Image out( width, height );

		Invert( in, startXIn, startYIn, out, 0, 0, width, height );

		return out;
	}

	void Invert( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
				  uint32_t width, uint32_t height )
	{
		FunctionTask().Invert(in, startXIn, startYIn, out, startXOut, startYOut, width, height );
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

	Image Maximum( const Image & in1, const Image & in2 )
	{
		Image_Function::ParameterValidation( in1, in2 );

		Image out( in1.width(), in1.height() );

		Maximum( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void Maximum( const Image & in1, const Image & in2, Image & out )
	{
		Image_Function::ParameterValidation( in1, in2, out );

		Maximum( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
	}

	Image Maximum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
				   uint32_t width, uint32_t height )
	{
		Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

		Image out( width, height );

		Maximum( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void Maximum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
				  Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
	{
		FunctionTask().Maximum(in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height);
	}

	Image Minimum( const Image & in1, const Image & in2 )
	{
		Image_Function::ParameterValidation( in1, in2 );

		Image out( in1.width(), in1.height() );

		Minimum( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void Minimum( const Image & in1, const Image & in2, Image & out )
	{
		Image_Function::ParameterValidation( in1, in2, out );

		Minimum( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
	}

	Image Minimum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
				   uint32_t width, uint32_t height )
	{
		Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

		Image out( width, height );

		Minimum( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void Minimum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
				  Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
	{
		FunctionTask().Minimum(in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height);
	}

	Image Normalize( const Image & in )
	{
		Image_Function::ParameterValidation( in );

		Image out( in.width(), in.height() );

		Normalize( in, 0, 0, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void Normalize( const Image & in, Image & out )
	{
		Image_Function::ParameterValidation( in, out );

		Normalize( in, 0, 0, out, 0, 0, out.width(), out.height() );
	}

	Image Normalize( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
	{
		Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

		Image out( width, height );

		Normalize( in, startXIn, startYIn, out, 0, 0, width, height );

		return out;
	}
	
	void Normalize( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
					uint32_t width, uint32_t height )
	{
		FunctionTask().Normalize(in, startXIn, startYIn, out, startXOut, startYOut, width, height );
	}

	std::vector < uint32_t > ProjectionProfile( const Image & image, bool horizontal )
	{
		std::vector < uint32_t > projection;

		ProjectionProfile( image, 0, 0, image.width(), image.height(), horizontal, projection );

		return projection;
	}

	void ProjectionProfile( const Image & image, bool horizontal, std::vector < uint32_t > & projection )
	{
		ProjectionProfile( image, 0, 0, image.width(), image.height(), horizontal, projection );
	}

	std::vector < uint32_t > ProjectionProfile( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, bool horizontal )
	{
		std::vector < uint32_t > projection;

		ProjectionProfile( image, x, y, width, height, horizontal, projection );

		return projection;
	}

	void ProjectionProfile( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, bool horizontal,
							std::vector < uint32_t > & projection )
	{
		FunctionTask().ProjectionProfile(image, x, y, width, height, horizontal, projection );
	}

	Image Resize( const Image & in, uint32_t widthOut, uint32_t heightOut )
	{
		Image_Function::ParameterValidation( in );

		Image out( widthOut, heightOut );

		Resize( in, 0, 0, in.width(), in.height(), out, 0, 0, widthOut, heightOut );

		return out;
	}

	void Resize( const Image & in, Image & out )
	{
		Image_Function::ParameterValidation( in );
		Image_Function::ParameterValidation( out );

		Resize( in, 0, 0, in.width(), in.height(), out, 0, 0, out.width(), out.height() );
	}

	Image Resize( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t widthIn, uint32_t heightIn,
				  uint32_t widthOut, uint32_t heightOut )
	{
		Image_Function::ParameterValidation( in, startXIn, startYIn, widthIn, heightIn );

		Image out( widthOut, heightOut );

		Resize( in, startXIn, startYIn, widthIn, heightIn, out, 0, 0, widthOut, heightOut );

		return out;
	}

	void Resize( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t widthIn, uint32_t heightIn,
				 Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t widthOut, uint32_t heightOut )
	{
		Image_Function::ParameterValidation( in,  startXIn,  startYIn,  widthIn,  heightIn );
		Image_Function::ParameterValidation( out, startXOut, startYOut, widthOut, heightOut );

		FunctionTask().Resize(in, startXIn, startYIn, widthIn, heightIn, out, startXOut, startYOut, widthOut, heightOut );
	}

	Image RgbToBgr(const Image & in)
	{
		Image_Function::ParameterValidation(in);

		Image out(in.width(), in.height(), 3u);

		RgbToBgr(in, 0, 0, out, 0, 0, in.width(), in.height());

		return out;
	}

	void RgbToBgr(const Image & in, Image & out)
	{
		Image_Function::ParameterValidation(in, out);

		RgbToBgr(in, 0, 0, out, 0, 0, in.width(), in.height());
	}

	Image RgbToBgr( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
	{
		Image_Function::ParameterValidation(in, startXIn, startYIn, width, height);

		Image out(width, height, 3u);

		RgbToBgr(in, startXIn, startYIn, out, 0, 0, width, height);

		return out;
	}

	void RgbToBgr( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
				   uint32_t width, uint32_t height )
	{
		FunctionTask().RgbToBgr(in, startXIn, startYIn, out, startXOut, startYOut, width, height );
	}

	Image Subtract( const Image & in1, const Image & in2 )
	{
		Image_Function::ParameterValidation( in1, in2 );

		Image out( in1.width(), in1.height() );

		Subtract( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void Subtract( const Image & in1, const Image & in2, Image & out )
	{
		Image_Function::ParameterValidation( in1, in2, out );

		Subtract( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
	}

	Image Subtract( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					uint32_t width, uint32_t height )
	{
		Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

		Image out( width, height );

		Subtract( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void Subtract( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
				   Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
	{
		FunctionTask().Subtract(in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height);
	}

	uint32_t Sum( const Image & image )
	{
		return Sum( image, 0, 0, image.width(), image.height() );
	}

	uint32_t Sum( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height )
	{
		return FunctionTask().Sum(image, x, y, width, height );
	}

	Image Threshold( const Image & in, uint8_t threshold )
	{
		Image_Function::ParameterValidation( in );

		Image out( in.width(), in.height() );

		Threshold( in, 0, 0, out, 0, 0, out.width(), out.height(), threshold );

		return out;
	}

	void Threshold( const Image & in, Image & out, uint8_t threshold )
	{
		Image_Function::ParameterValidation( in, out );

		Threshold( in, 0, 0, out, 0, 0, out.width(), out.height(), threshold );
	}

	Image Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, uint8_t threshold )
	{
		Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

		Image out( width, height );

		Threshold( in, startXIn, startYIn, out, 0, 0, width, height, threshold );

		return out;
	}

	void Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
					uint32_t width, uint32_t height, uint8_t threshold )
	{
		FunctionTask().Threshold(in, startXIn, startYIn, out, startXOut, startYOut, width, height, threshold );
	}

	Image Threshold( const Image & in, uint8_t minThreshold, uint8_t maxThreshold )
	{
		Image_Function::ParameterValidation( in );

		Image out( in.width(), in.height() );

		Threshold( in, 0, 0, out, 0, 0, out.width(), out.height(), minThreshold, maxThreshold );

		return out;
	}

	void Threshold( const Image & in, Image & out, uint8_t minThreshold, uint8_t maxThreshold )
	{
		Image_Function::ParameterValidation( in, out );

		Threshold( in, 0, 0, out, 0, 0, out.width(), out.height(), minThreshold, maxThreshold );
	}

	Image Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, uint8_t minThreshold,
					 uint8_t maxThreshold )
	{
		Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

		Image out( width, height );

		Threshold( in, startXIn, startYIn, out, 0, 0, width, height, minThreshold, maxThreshold );

		return out;
	}

	void Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
					uint32_t width, uint32_t height, uint8_t minThreshold, uint8_t maxThreshold )
	{
		FunctionTask().Threshold(in, startXIn, startYIn, out, startXOut, startYOut, width, height, minThreshold, maxThreshold );
	}
};
