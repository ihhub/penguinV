/***************************************************************************
 *   penguinV: https://github.com/ihhub/penguinV                           *
 *   Copyright (C) 2017 - 2024                                             *
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

#pragma once
#include "image_buffer.h"
#include "math/math_base.h"
#include <vector>

namespace Blob_Detection
{
    template <typename Data>
    struct Parameter
    {
        // Minimum value.
        Data minimum{ 0 };

        // Whether the value should be compared with the minimum value.
        bool checkMinimum{ false };

        // Maximum value.
        Data maximum{ 0 };

        // Whether the value should be compared with the maximum value.
        bool checkMaximum{ false };

        void verify() const
        {
            if ( checkMaximum && checkMinimum && ( minimum > maximum ) ) {
                throw penguinVException( "Minimum value cannot be bigger than maximum value" );
            }
        }

        void set( const Data min, const Data max )
        {
            setMinimum( min );
            setMaximum( max );
        }

        void setMinimum( const Data min )
        {
            checkMinimum = true;
            minimum = min;
        }

        void setMaximum( const Data max )
        {
            checkMaximum = true;
            maximum = max;
        }

        void reset()
        {
            *this = {};
        }
    };

    struct BlobParameters
    {
        // This parameter will be 1 if blob is ideal circle and will be less than 1 if it's not,
        // Closer this value to 1 --> blob shape is closer to circle,
        Parameter<double> circularity;

        // Some people call it inertia: ratio of the minor and major axes of a blob.
        Parameter<double> elongation;

        // Width, in pixels.
        Parameter<uint32_t> width;

        // Height, in pixels.
        Parameter<uint32_t> height;

        // Maximum distance between any of 2 pixels, in pixels.
        Parameter<double> length;

        // Overall size of blobs, in pixels.
        Parameter<uint32_t> size;

        // this function will be called in BlobInfo class before finding blobs
        void _verify() const
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

    struct Area
    {
        Area() = default;

        // this constructor is made to avoid 'Value' template restriction
        explicit Area( uint32_t value )
            : left( value )
            , right( value )
            , top( value )
            , bottom( value )
        {
            // Do nothing.
        }

        uint32_t left{ 0 };
        uint32_t right{ 0 };
        uint32_t top{ 0 };
        uint32_t bottom{ 0 };
    };

    template <typename Data>
    struct Value
    {
        Data value;
        bool found{ false };
    };

    // This class follows an idea of lazy computations:
    // calculate result when it is needed. If you forgot to specify some parameter
    // in BlobParameter structure for evaluation you can still retrieve the value after.
    // But sorting of blobs in BlobDetection class will depend on input BlobParameter parameters so set proper parameters ;)
    class BlobInfo
    {
    public:
        friend class BlobDetection;

        // Returns an array what contains all blob's pixel X positions (unsorted).
        const std::vector<uint32_t> & pointX() const
        {
            return _pointX;
        }

        // Returns an array what contains all blob's pixel Y positions (unsorted).
        const std::vector<uint32_t> & pointY() const
        {
            return _pointY;
        }

        // Returns an array what contains all blob's contour pixel X positions (unsorted).
        const std::vector<uint32_t> & contourX() const
        {
            return _contourX;
        }

        // Returns an array what contains all blob's contour pixel Y positions (unsorted).
        const std::vector<uint32_t> & contourY() const
        {
            return _contourY;
        }

        // Returns an array what contains all blob's edge pixel X positions (unsorted).
        const std::vector<uint32_t> & edgeX() const
        {
            return _edgeX;
        }

        // Returns an array what contains all blob's edge pixel Y positions (unsorted).
        const std::vector<uint32_t> & edgeY() const
        {
            return _edgeY;
        }

        // Each function has 2 overloaded forms:
        // - non-constant function check whether value was calculated, calculates it, if necessary, and return value
        // - constant function return value no matter a value was calculated or not.

        // Minimum fitting rectangle what can contain the blob.
        Area area()
        {
            _getArea();

            return _area.value;
        }

        // Minimum fitting rectangle what can contain the blob.
        Area area() const
        {
            return _area.value;
        }

        // Gravity center of the blob.
        Point2d center()
        {
            _getCenter();

            return _center.value;
        }

        // Gravity center of the blob.
        Point2d center() const
        {
            return _center.value;
        }

        // Circularity of the blob.
        double circularity()
        {
            _getCircularity();

            return _circularity.value;
        }

        // Circularity of the blob.
        double circularity() const
        {
            return _circularity.value;
        }

        // Elongation of the blob.
        double elongation()
        {
            _getElongation();

            return _elongation.value;
        }

        // Elongation of the blob.
        double elongation() const
        {
            return _elongation.value;
        }

        // Width of the blob.
        uint32_t width()
        {
            _getWidth();

            return _width.value;
        }

        // Width of the blob.
        uint32_t width() const
        {
            return _width.value;
        }

        // Height of the blob.
        uint32_t height()
        {
            _getHeight();

            return _height.value;
        }

        // Height of the blob.
        uint32_t height() const
        {
            return _height.value;
        }

        // Length of the blob.
        double length()
        {
            _getLength();

            return _length.value;
        }

        // Length of the blob.
        double length() const
        {
            return _length.value;
        }

        // Total number of pixels in the blob.
        size_t size() const
        {
            return _pointX.size();
        }

        // Returns true if blob does not have inner edge points.
        bool isSolid() const
        {
            return _contourX.size() == _edgeX.size();
        }

    private:
        std::vector<uint32_t> _pointX;
        std::vector<uint32_t> _pointY;
        std::vector<uint32_t> _contourX;
        std::vector<uint32_t> _contourY;
        std::vector<uint32_t> _edgeX;
        std::vector<uint32_t> _edgeY;

        Value<Area> _area;
        Value<Point2d> _center;
        Value<double> _circularity;
        Value<double> _elongation;
        Value<uint32_t> _height;
        Value<double> _length;
        Value<uint32_t> _width;

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
        const std::vector<BlobInfo> & find( const penguinV::Image & image, const BlobParameters & parameter = BlobParameters(), uint8_t threshold = 1 )
        {
            return find( image, 0, 0, image.width(), image.height(), parameter, threshold );
        }

        const std::vector<BlobInfo> & find( const penguinV::Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
                                            const BlobParameters & parameter = BlobParameters(), uint8_t threshold = 1 );

        // Retrieve an array of all found blobs
        const std::vector<BlobInfo> & get() const
        {
            return _blob;
        }

        std::vector<BlobInfo> & get()
        {
            return _blob;
        }

        // These are same functions, added to simplify coding.
        const std::vector<BlobInfo> & operator()() const
        {
            return _blob;
        }

        std::vector<BlobInfo> & operator()()
        {
            return _blob;
        }

        enum class BlobCriterion : uint8_t
        {
            BY_CIRCULARITY,
            BY_ELONGATION,
            BY_HEIGHT,
            BY_LENGTH,
            BY_SIZE,
            BY_WIDTH
        };

        // before calling this function make sure that you have more than 1 found blob!
        const BlobInfo & getBestBlob( BlobCriterion criterion ) const;
        // sorting all found blobs in ascending order
        void sort( BlobCriterion criterion );

    protected:
        std::vector<BlobInfo> _blob;
    };
}
