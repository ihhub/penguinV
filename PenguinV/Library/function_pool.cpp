#include "function_pool.h"
#include "image_function.h"
#include "thread_pool.h"

namespace Function_Pool
{
	struct AreaInfo
	{
		AreaInfo( uint32_t x, uint32_t y, uint32_t width_, uint32_t height_, uint32_t count )
		{
			_calculate( x, y, width_, height_, count );
		};

		std::vector < uint32_t > startX; // start X position of image roi
		std::vector < uint32_t > startY; // start Y position of image roi
		std::vector < uint32_t > width;  // width of image roi
		std::vector < uint32_t > height; // height of image roi

		size_t _size() const
		{
			return startX.size();
		}
	private:
		static const uint32_t cacheSize = 16; // Remember: every CPU has it's own caching technique so processing time of subsequent memory cells is much faster!
											  // Change this value if you need to adjust to specific CPU. 16 bytes are set for proper SSE support

		// this function will sort out all input data into arrays for multithreading execution
		void _calculate( uint32_t x, uint32_t y, uint32_t width_, uint32_t height_, uint32_t count )
		{
			uint32_t maximumXTaskCount = width_  / cacheSize;
			if( maximumXTaskCount > count )
				maximumXTaskCount = count;

			uint32_t maximumYTaskCount = height_ / cacheSize;
			if( maximumYTaskCount > count )
				maximumYTaskCount = count;

			if( maximumYTaskCount >= maximumXTaskCount ) { // process by rows
				count = maximumYTaskCount;

				startX.resize( count );
				startY.resize( count );
				width .resize( count );
				height.resize( count );

				std::fill( startX.begin(), startX.end(), x );
				std::fill( width.begin() , width.end() , width_ );

				uint32_t remainValue = height_ % count;
				uint32_t previousValue = 0;

				for( size_t i = 0; i < startX.size(); ++i ) {
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
				count = maximumXTaskCount;

				startX.resize( count );
				startY.resize( count );
				width .resize( count );
				height.resize( count );

				std::fill( startY.begin(), startY.end(), y );
				std::fill( height.begin(), height.end(), height_ );

				uint32_t remainValue = width_ % count;
				uint32_t previousValue = 0;

				for( size_t i = 0; i < startX.size(); ++i ) {
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

	struct InputImageInfo : AreaInfo
	{
		InputImageInfo(const Image & in, uint32_t x, uint32_t y, uint32_t width_, uint32_t height_, uint32_t count)
			: AreaInfo( x, y, width_, height_, count)
			, image(in)
		{ }

		const Image & image;
	};

	struct OutputImageInfo : AreaInfo
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
			: threshold           (0)
			, horizontalProjection(false)
		{ }

		uint8_t threshold; // for Threshold() function
		bool horizontalProjection; // for ProjectionProfile() function
	};
	// This structure holds output data for some specific functions
	struct OutputInfo
	{
		std::vector < std::vector < uint32_t > > histogram;  // for Histogram() function
		std::vector < std::vector < uint32_t > > projection; // for ProjectionProfile() function

		void resize(size_t count)
		{
			histogram.resize(count);
		}

		void getHistogram(std::vector <uint32_t> & histogram_ )
		{
			_getArray( histogram, histogram_ );
		}

		void getProjection(std::vector <uint32_t> & projection_ )
		{
			_getArray( projection, projection_ );
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

	class FunctionTask : Thread_Pool::TaskProviderSingleton
	{
	public:
		FunctionTask()
			: functionId(_none)
		{ };

		// this is a list of image functions
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

		void Histogram( const Image & image, uint32_t x, int32_t y, uint32_t width, uint32_t height,
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

		void ProjectionProfile( const Image & image, uint32_t x, int32_t y, uint32_t width, uint32_t height, bool horizontal,
								std::vector < uint32_t > & projection )
		{
			_setup( image, x, y, width, height );
			_dataOut.resize(_infoIn1->_size());
			_dataIn.horizontalProjection = horizontal;
			_process( _ProjectionProfile );
			_dataOut.getProjection( projection );
		}

		void Subtract( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
						 Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
		{
			_setup( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
			_process( _Subtract );
		}

		void Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
						uint32_t width, uint32_t height, uint8_t threshold )
		{
			_setup( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
			_dataIn.threshold = threshold;
			_process( _Threshold );
		}
	protected:
		enum TaskName // enumeration to define for thread what function need to execute
		{
			_none,
			_BitwiseAnd,
			_BitwiseOr,
			_BitwiseXor,
			_Histogram,
			_Invert,
			_Maximum,
			_Minimum,
			_Normalize,
			_ProjectionProfile,
			_Subtract,
			_Threshold
		};

		void _task(size_t taskId)
		{
			switch(functionId) {
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
			case _Histogram:
				Image_Function::Histogram(  _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
											_infoIn1->width[taskId], _infoIn1->height[taskId], _dataOut.histogram[taskId] );
				break;
			case _Invert:
				Image_Function::Invert(     _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
											_infoOut->image, _infoOut->startX[taskId], _infoOut->startY[taskId],
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
				Image_Function::ProjectionProfile(  _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
											_infoIn1->width[taskId], _infoIn1->height[taskId], _dataIn.horizontalProjection,
											_dataOut.projection[taskId] );
				break;
			case _Subtract:
				Image_Function::Subtract(   _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
											_infoIn2->image, _infoIn2->startX[taskId], _infoIn2->startY[taskId],
											_infoOut->image, _infoOut->startX[taskId], _infoOut->startY[taskId],
											_infoIn1->width[taskId], _infoIn1->height[taskId] );
				break;
			case _Threshold:
				Image_Function::Threshold(  _infoIn1->image, _infoIn1->startX[taskId], _infoIn1->startY[taskId],
											_infoOut->image, _infoOut->startX[taskId], _infoOut->startY[taskId],
											_infoIn1->width[taskId], _infoIn1->height[taskId], _dataIn.threshold );
				break;
			default:
				throw imageException("Wrong image function task");
			}
		}

	private:
		TaskName functionId;
		std::unique_ptr < InputImageInfo  > _infoIn1; // structure that holds information about first input image
		std::unique_ptr < InputImageInfo  > _infoIn2; // structure that holds information about second input image
		std::unique_ptr < OutputImageInfo > _infoOut; // structure that holds information about output image

		InputInfo  _dataIn;  // structure that hold some unique input parameters
		OutputInfo _dataOut; // structure that hold some unique output values

		// functions for setting up all parameters needed for multithreading and to validate input parameters
		void _setup( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height )
		{
			Image_Function::ParameterValidation( image, x, y, width, height );

			if( !_ready() )
				throw imageException("FunctionTask object was called multiple times!");

			uint32_t count = static_cast<uint32_t>(Thread_Pool::ThreadPoolMonoid::instance().threadCount());

			_infoIn1 = std::unique_ptr < InputImageInfo  >( new InputImageInfo ( image, x  , y  , width, height, count ) );
		}

		void _setup( const Image & in, uint32_t inX, uint32_t inY, Image & out, uint32_t outX, uint32_t outY, uint32_t width, uint32_t height )
		{
			Image_Function::ParameterValidation( in, inX, inY, out, outX, outY, width, height );

			if( !_ready() )
				throw imageException("FunctionTask object was called multiple times!");

			uint32_t count = static_cast<uint32_t>(Thread_Pool::ThreadPoolMonoid::instance().threadCount());

			_infoIn1 = std::unique_ptr < InputImageInfo  >( new InputImageInfo ( in, inX , inY  , width, height, count ) );
			_infoOut = std::unique_ptr < OutputImageInfo >( new OutputImageInfo( out, outX, outY, width, height, count ) );
		}

		void _setup( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					 Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
		{
			Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );

			if( !_ready() )
				throw imageException("FunctionTask object was called multiple times!");

			uint32_t count = static_cast<uint32_t>(Thread_Pool::ThreadPoolMonoid::instance().threadCount());

			_infoIn1 = std::unique_ptr < InputImageInfo  >( new InputImageInfo ( in1, startX1  , startY1  , width, height, count ) );
			_infoIn2 = std::unique_ptr < InputImageInfo  >( new InputImageInfo ( in2, startX2  , startY2  , width, height, count ) );
			_infoOut = std::unique_ptr < OutputImageInfo >( new OutputImageInfo( out, startXOut, startYOut, width, height, count ) );
		}

		void _process(TaskName id) // function what calls global thread pool and waits results from it
		{
			functionId = id;

			_run( _infoIn1->height.size() );

			_wait();
		}
	};

	// The list of global functions
	void BitwiseAnd( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					 Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
	{
		std::unique_ptr < FunctionTask > ptr( new FunctionTask );

		ptr->BitwiseAnd(in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height);
	}

	Image BitwiseAnd( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
				  uint32_t width, uint32_t height )
	{
		Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

		Image out( width, height );

		BitwiseAnd( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void BitwiseAnd( const Image & in1, const Image & in2, Image & out )
	{
		Image_Function::ParameterValidation( in1, in2, out );

		BitwiseAnd( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
	}

	Image BitwiseAnd( const Image & in1, const Image & in2 )
	{
		Image_Function::ParameterValidation( in1, in2 );

		Image out( in1.width(), in1.height() );

		BitwiseAnd( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void BitwiseOr( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					 Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
	{
		std::unique_ptr < FunctionTask > ptr( new FunctionTask );

		ptr->BitwiseOr(in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height);
	}

	Image BitwiseOr( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
				  uint32_t width, uint32_t height )
	{
		Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

		Image out( width, height );

		BitwiseOr( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void BitwiseOr( const Image & in1, const Image & in2, Image & out )
	{
		Image_Function::ParameterValidation( in1, in2, out );

		BitwiseOr( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
	}

	Image BitwiseOr( const Image & in1, const Image & in2 )
	{
		Image_Function::ParameterValidation( in1, in2 );

		Image out( in1.width(), in1.height() );

		BitwiseOr( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void BitwiseXor( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					 Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
	{
		std::unique_ptr < FunctionTask > ptr( new FunctionTask );

		ptr->BitwiseXor(in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height);
	}

	Image BitwiseXor( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
				  uint32_t width, uint32_t height )
	{
		Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

		Image out( width, height );

		BitwiseXor( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void BitwiseXor( const Image & in1, const Image & in2, Image & out )
	{
		Image_Function::ParameterValidation( in1, in2, out );

		BitwiseXor( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
	}

	Image BitwiseXor( const Image & in1, const Image & in2 )
	{
		Image_Function::ParameterValidation( in1, in2 );

		Image out( in1.width(), in1.height() );

		BitwiseXor( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

		return out;
	}

	uint8_t GetThreshold( const Image & image )
	{
		uint8_t threshold;

		GetThreshold( image, 0, 0, image.width(), image.height(), threshold );

		return threshold;
	}

	void GetThreshold( const Image & image, uint8_t & threshold )
	{
		GetThreshold( image, 0, 0, image.width(), image.height(), threshold );
	}
	
	uint8_t GetThreshold( const Image & image, uint32_t x, int32_t y, uint32_t width, uint32_t height )
	{
		uint8_t threshold;

		GetThreshold( image, x, y, width, height, threshold );

		return threshold;
	}

	void GetThreshold( const Image & image, uint32_t x, int32_t y, uint32_t width, uint32_t height, uint8_t & threshold )
	{
		std::vector < uint32_t > histogram = Histogram( image, x, y, width, height );

		threshold = 0;

		// It is well-known Otsu's method to find threshold
		uint32_t sum = histogram[1];
		for(uint16_t i = 2; i < 256; ++i)
			sum  = sum  + i * histogram[i];

		uint32_t sumTemp = 0;

		uint32_t pixelCount     = width * height;
		uint32_t pixelCountTemp = 0;
		
		double maximumSigma = -1;

		for(uint16_t i = 0; i < 256; ++i) {

			pixelCountTemp += histogram[i];

			if(pixelCountTemp > 0 && pixelCountTemp != pixelCount) {
				sumTemp += i * histogram[i];

				double w1 = static_cast<double>(pixelCountTemp) / pixelCount;
				double a  = static_cast<double>(sumTemp       ) / pixelCountTemp -
						    static_cast<double>(sum - sumTemp ) / (pixelCount - pixelCountTemp);
				double sigma = w1 * (1 - w1) * a * a;

				if(sigma > maximumSigma) {
					maximumSigma = sigma;
					threshold = static_cast < uint8_t >(i);
				}

			}

		}
	}

	void Histogram( const Image & image, uint32_t x, int32_t y, uint32_t width, uint32_t height, std::vector < uint32_t > & histogram )
	{
		std::unique_ptr < FunctionTask > ptr( new FunctionTask );

		ptr->Histogram(image, x, y, width, height, histogram );
	}

	std::vector < uint32_t > Histogram( const Image & image, uint32_t x, int32_t y, uint32_t width, uint32_t height )
	{
		Image_Function::ParameterValidation( image, x, y, width, height );

		std::vector < uint32_t > histogram;

		Histogram( image, x, y, width, height, histogram );

		return histogram;
	}

	void Histogram( const Image & image, std::vector < uint32_t > & histogram )
	{
		Histogram( image, 0, 0, image.width(), image.height(), histogram );
	}

	std::vector < uint32_t > Histogram( const Image & image )
	{
		std::vector < uint32_t > histogram;

		Histogram( image, 0, 0, image.width(), image.height(), histogram );

		return histogram;
	}

	void  Invert( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
				  uint32_t width, uint32_t height )
	{
		std::unique_ptr < FunctionTask > ptr( new FunctionTask );

		ptr->Invert(in, startXIn, startYIn, out, startXOut, startYOut, width, height );
	}

	Image Invert( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
	{
		Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

		Image out( width, height );

		Invert( in, startXIn, startYIn, out, 0, 0, width, height );

		return out;
	}

	void  Invert( const Image & in, Image & out )
	{
		Image_Function::ParameterValidation( in, out );

		Invert( in, 0, 0, out, 0, 0, out.width(), out.height() );
	}

	Image Invert( const Image & in )
	{
		Image_Function::ParameterValidation( in );

		Image out( in.width(), in.height() );

		Invert( in, 0, 0, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void Maximum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
				  Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
	{
		std::unique_ptr < FunctionTask > ptr( new FunctionTask );

		ptr->Maximum(in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height);
	}

	Image Maximum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
				   uint32_t width, uint32_t height )
	{
		Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

		Image out( width, height );

		Maximum( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void Maximum( const Image & in1, const Image & in2, Image & out )
	{
		Image_Function::ParameterValidation( in1, in2, out );

		Maximum( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
	}

	Image Maximum( const Image & in1, const Image & in2 )
	{
		Image_Function::ParameterValidation( in1, in2 );

		Image out( in1.width(), in1.height() );

		Maximum( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void Minimum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
				  Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
	{
		std::unique_ptr < FunctionTask > ptr( new FunctionTask );

		ptr->Minimum(in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height);
	}

	Image Minimum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
				   uint32_t width, uint32_t height )
	{
		Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

		Image out( width, height );

		Minimum( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void Minimum( const Image & in1, const Image & in2, Image & out )
	{
		Image_Function::ParameterValidation( in1, in2, out );

		Minimum( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
	}

	Image Minimum( const Image & in1, const Image & in2 )
	{
		Image_Function::ParameterValidation( in1, in2 );

		Image out( in1.width(), in1.height() );

		Minimum( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

		return out;
	}
	
	void  Normalize( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
					 uint32_t width, uint32_t height )
	{
		std::unique_ptr < FunctionTask > ptr( new FunctionTask );

		ptr->Normalize(in, startXIn, startYIn, out, startXOut, startYOut, width, height );
	}

	Image Normalize( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
	{
		Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

		Image out( width, height );

		Normalize( in, startXIn, startYIn, out, 0, 0, width, height );

		return out;
	}

	void  Normalize( const Image & in, Image & out )
	{
		Image_Function::ParameterValidation( in, out );

		Normalize( in, 0, 0, out, 0, 0, out.width(), out.height() );
	}

	Image Normalize( const Image & in )
	{
		Image_Function::ParameterValidation( in );

		Image out( in.width(), in.height() );

		Normalize( in, 0, 0, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void ProjectionProfile( const Image & image, uint32_t x, int32_t y, uint32_t width, uint32_t height, bool horizontal,
							std::vector < uint32_t > & projection )
	{
		std::unique_ptr < FunctionTask > ptr( new FunctionTask );

		ptr->ProjectionProfile(image, x, y, width, height, horizontal, projection );
	}

	std::vector < uint32_t > ProjectionProfile( const Image & image, uint32_t x, int32_t y, uint32_t width, uint32_t height, bool horizontal )
	{
		std::vector < uint32_t > projection;

		ProjectionProfile( image, x, y, width, height, horizontal, projection );

		return projection;
	}

	void ProjectionProfile( const Image & image, bool horizontal, std::vector < uint32_t > & projection )
	{
		ProjectionProfile( image, 0, 0, image.width(), image.height(), horizontal, projection );
	}

	std::vector < uint32_t > ProjectionProfile( const Image & image, bool horizontal )
	{
		std::vector < uint32_t > projection;

		ProjectionProfile( image, 0, 0, image.width(), image.height(), horizontal, projection );

		return projection;
	}

	void Subtract( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
				   Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
	{
		std::unique_ptr < FunctionTask > ptr( new FunctionTask );

		ptr->Subtract(in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height);
	}

	Image Subtract( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					uint32_t width, uint32_t height )
	{
		Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

		Image out( width, height );

		Subtract( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void Subtract( const Image & in1, const Image & in2, Image & out )
	{
		Image_Function::ParameterValidation( in1, in2, out );

		Subtract( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
	}

	Image Subtract( const Image & in1, const Image & in2 )
	{
		Image_Function::ParameterValidation( in1, in2 );

		Image out( in1.width(), in1.height() );

		Subtract( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
					uint32_t width, uint32_t height, uint8_t threshold )
	{
		std::unique_ptr < FunctionTask > ptr( new FunctionTask );

		ptr->Threshold(in, startXIn, startYIn, out, startXOut, startYOut, width, height, threshold );
	}

	Image Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, uint8_t threshold )
	{
		Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

		Image out( width, height );

		Threshold( in, startXIn, startYIn, out, 0, 0, width, height, threshold );

		return out;
	}

	void Threshold( const Image & in, Image & out, uint8_t threshold )
	{
		Image_Function::ParameterValidation( in, out );

		Threshold( in, 0, 0, out, 0, 0, out.width(), out.height(), threshold );
	}

	Image Threshold( const Image & in, uint8_t threshold )
	{
		Image_Function::ParameterValidation( in );

		Image out( in.width(), in.height() );

		Threshold( in, 0, 0, out, 0, 0, out.width(), out.height(), threshold );

		return out;
	}
};
