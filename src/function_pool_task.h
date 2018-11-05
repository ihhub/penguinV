#pragma once

#include <vector>
#include "image_buffer.h"
#include "thread_pool.h"

namespace Function_Pool
{
    using namespace PenguinV_Image;

    struct AreaInfo
    {
        AreaInfo( uint32_t x, uint32_t y, uint32_t width_, uint32_t height_, uint32_t count );

        std::vector < uint32_t > startX; // start X position of image ROI
        std::vector < uint32_t > startY; // start Y position of image ROI
        std::vector < uint32_t > width;  // width of image ROI
        std::vector < uint32_t > height; // height of image ROI

        size_t _size() const;

        // this function makes a similar input data sorting like it is done in info parameter
        void _copy( const AreaInfo & info, uint32_t x, uint32_t y, uint32_t width_, uint32_t height_ );
    private:
        // this function will sort out all input data into arrays for multithreading execution
        void _calculate( uint32_t x, uint32_t y, uint32_t width_, uint32_t height_, uint32_t count );

        // this function fills all arrays by necessary values
        void _fill( uint32_t x, uint32_t y, uint32_t width_, uint32_t height_, uint32_t count, bool yAxis );
    };

    struct InputImageInfo : public AreaInfo
    {
        InputImageInfo( const Image & in, uint32_t x, uint32_t y, uint32_t width_, uint32_t height_, uint32_t count );

        const Image & image;
    };

    struct OutputImageInfo : public AreaInfo
    {
        OutputImageInfo( Image & in, uint32_t x, uint32_t y, uint32_t width_, uint32_t height_, uint32_t count );

        Image & image;
    };

    class FunctionPoolTask : public TaskProviderSingleton
    {
    public:
        FunctionPoolTask();
        virtual ~FunctionPoolTask();
    protected:
        std::unique_ptr < InputImageInfo  > _infoIn1; // structure which holds information about first input image
        std::unique_ptr < InputImageInfo  > _infoIn2; // structure which holds information about second input image
        std::unique_ptr < OutputImageInfo > _infoOut; // structure which holds information about output image

        // functions for setting up all parameters needed for multithreading and to validate input parameters
        void _setup( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height );

        void _setup( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                     uint32_t width, uint32_t height );

        void _setup( const Image & in, uint32_t inX, uint32_t inY, Image & out, uint32_t outX, uint32_t outY, uint32_t width, uint32_t height );

        void _setup( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t widthIn, uint32_t heightIn,
                     Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t widthOut, uint32_t heightOut );

        void _setup( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                     Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height );

        virtual void _task( size_t taskId ) = 0;

        void _processTask(); // function which calls global thread pool and waits results from it
    };
}
