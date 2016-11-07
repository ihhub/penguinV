#pragma once
#include <vector>
#include "image_buffer.h"

namespace Blob_Detection
{
	template <typename Data>
	struct Parameter
	{
		Parameter()
			: minimum     (0)     // minimum value
			, checkMinimum(false) // set to compare a value with minimum value
			, maximum     (0)     // maximum value
			, checkMaximum(0)     // set to compare a value with maximum value
		{ };

		Data minimum;
		bool checkMinimum;

		Data maximum;
		bool checkMaximum;

		void verify()
		{
			if( minimum > maximum ) {
				std::swap( minimum, maximum );
				std::swap( checkMinimum, checkMaximum );
			}
		}

		void set( Data min, Data max )
		{
			setMinimum(min);
			setMaximum(max);
		}

		void setMinimum( Data min )
		{
			checkMinimum = true;
			minimum      = min;
		}

		void setMaximum( Data max )
		{
			checkMaximum = true;
			maximum      = max;
		}

		void reset()
		{
			*this = Parameter();
		}
	};

	struct BlobParameters
	{
		Parameter < double   > circularity; // this parameter will be 1 if blob is ideal circle and will be less than 1 if it's not
											// closer this value to 1 --> blob shape is closer to circle
		Parameter < double   > elongation;  // some people call it inertia: ratio of the minor and major axes of a blob
		Parameter < uint32_t > height;      // height, in pixels
		Parameter < double   > length;      // maximum distance between any of 2 pixels, in pixels
		Parameter < uint32_t > size;        // overall size of blobs, in pixels
		Parameter < uint32_t > width;       // width, in pixels

		// this function will be called in BlobInfo class before finding blobs
		void _verify()
		{
			circularity.verify();
			elongation.verify();
			height.verify();
			length.verify();
			size.verify();
			width.verify();
		}
		// reset all parameters to initial values
		void _reset()
		{
			circularity.reset();
			elongation.reset();
			height.reset();
			length.reset();
			size.reset();
			width.reset();
		}
	};

	struct Point
	{
		Point()
			: x(0)
			, y(0)
		{ };

		// this constructor is made to avoid 'Value' template restriction
		Point(double value)
			: x(value)
			, y(value)
		{ };

		double x;
		double y;
	};

	struct Area
	{
		Area()
			: left  (0)
			, right (0)
			, top   (0)
			, bottom(0)
		{ };

		// this constructor is made to avoid 'Value' template restriction
		Area(uint32_t value)
			: left  (value)
			, right (value)
			, top   (value)
			, bottom(value)
		{ };

		uint32_t left;
		uint32_t right;
		uint32_t top;
		uint32_t bottom;
	};

	template <typename Data>
	struct Value
	{
		Value()
			: value(0)
			, found(false)
		{ };

		Data value;
		bool found;
	};

	// This class follows an idea of lazy computations:
	// calculate result when it is needed. If you forgot to specify some parameter
	// in BlobParameter structure for evaluation you can still retrieve the value after.
	// But sorting of blobs in BlobDetection class will depend on input BlobParameter parameters so set proper parameters ;)
	class BlobInfo
	{
	public:
		friend class BlobDetection;

		const std::vector < uint32_t > & pointX() const;   // returns an array what contains all blob's pixel X positions (unsorted)
		const std::vector < uint32_t > & pointY() const;   // returns an array what contains all blob's pixel Y positions (unsorted)
		const std::vector < uint32_t > & contourX() const; // returns an array what contains all blob's contour pixel X positions (unsorted)
		const std::vector < uint32_t > & contourY() const; // returns an array what contains all blob's contour pixel Y positions (unsorted)
		const std::vector < uint32_t > & edgeX() const;    // returns an array what contains all blob's edge pixel X positions (unsorted)
		const std::vector < uint32_t > & edgeY() const;    // returns an array what contains all blob's edge pixel Y positions (unsorted)

		// Each function has 2 overloaded forms:
		// - non-constant function check whether value was calculated, calculates it if neccessary and return value
		// - constant function return value no matter a value was calculated or not
		Area     area();              // minimum fitting rectangle what can contain blob
		Area     area() const;        // minimum fitting rectangle what can contain blob
		Point    center();            // gravity center of blob
		Point    center() const;      // gravity center of blob
		double   circularity();       // circularity of blob
		double   circularity() const; // circularity of blob
		double   elongation();        // elongation of blob
		double   elongation() const;  // elongation of blob
		uint32_t height();            // height of blob
		uint32_t height() const;      // height of blob
		double   length();            // length of blob
		double   length() const;      // length of blob
		size_t   size() const;        // total number of pixels in blob
		uint32_t width();             // width of blob
		uint32_t width() const;       // width of blob

		bool isSolid() const;         // true if blob does not have inner edge points
	private:
		std::vector < uint32_t > _pointX;
		std::vector < uint32_t > _pointY;
		std::vector < uint32_t > _contourX;
		std::vector < uint32_t > _contourY;
		std::vector < uint32_t > _edgeX;
		std::vector < uint32_t > _edgeY;

		Value < Area   > _area;
		Value < Point  > _center;
		Value < double > _circularity;
		Value < double > _elongation;
		Value <uint32_t> _height;
		Value < double > _length;
		Value <uint32_t> _width;

		void _getArea();
		void _getCenter();
		void _getCircularity();
		void _getElongation();
		void _getHeight();
		void _getLength();
		void _getWidth();
	};

	class BlobDetection
	{
	public:
		// Sorting blobs will be in alphabet order of sorting criteria
		// Example: length and width criteria enabled. So first all blobs would be removed if they are not fitting length criterion
		// and then all remain blobs would be removed if they are not fitting for width criterion
		const std::vector < BlobInfo > & find( const Bitmap_Image::Image & image, BlobParameters parameter = BlobParameters(),
											   uint8_t threshold = 1 );
		const std::vector < BlobInfo > & find( const Bitmap_Image::Image & image, uint32_t x, int32_t y, uint32_t width, uint32_t height,
											   BlobParameters parameter = BlobParameters(), uint8_t threshold = 1 );

		// Retrieve an array of all found blobs
		const std::vector < BlobInfo > & get() const;
		      std::vector < BlobInfo > & get();
		const std::vector < BlobInfo > & operator()() const; // these are same functions, added to simplify coding
		      std::vector < BlobInfo > & operator()();

		enum BlobCriterion
		{
			CRITERION_CIRCULARITY,
			CRITERION_ELONGATION,
			CRITERION_HEIGHT,
			CRITERION_LENGTH,
			CRITERION_SIZE,
			CRITERION_WIDTH
		};

		// before calling this function make sure that you have more than 1 found blob!
		const BlobInfo & getBestBlob(BlobCriterion criterion) const;
		// sorting all found blobs in ascending order
		void sort(BlobCriterion criterion);
	protected:
		std::vector < BlobInfo > _blob;
	};
};
