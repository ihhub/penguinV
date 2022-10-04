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

#pragma once

#include "image_buffer.h"
#include "thread_pool.h"
#include <vector>

namespace Function_Pool
{
    using namespace penguinV;

    struct AreaInfo
    {
        AreaInfo( uint32_t x, uint32_t y, uint32_t width_, uint32_t height_, uint32_t count );

        std::vector<uint32_t> startX; // start X position of image ROI
        std::vector<uint32_t> startY; // start Y position of image ROI
        std::vector<uint32_t> width; // width of image ROI
        std::vector<uint32_t> height; // height of image ROI

        size_t _size() const;

        // makes a similar input data sorting like it is done in info parameter
        void _copy( const AreaInfo & info, uint32_t x, uint32_t y, uint32_t width_, uint32_t height_, bool oppositeAxis = false );

    private:
        // sorts out all input data into arrays for multithreading execution
        void _calculate( uint32_t x, uint32_t y, uint32_t width_, uint32_t height_, uint32_t count );

        // fills all arrays by necessary values
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
        // input images
        std::unique_ptr<InputImageInfo> _infoIn1;
        std::unique_ptr<InputImageInfo> _infoIn2;
        std::unique_ptr<InputImageInfo> _infoIn3;
        // output images
        std::unique_ptr<OutputImageInfo> _infoOut1;
        std::unique_ptr<OutputImageInfo> _infoOut2;
        std::unique_ptr<OutputImageInfo> _infoOut3;

        // functions for setting up all parameters needed for multithreading and to validate input parameters
        void _setup( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height );

        void _setup( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2, uint32_t width, uint32_t height );

        void _setup( const Image & in, uint32_t inX, uint32_t inY, Image & out, uint32_t outX, uint32_t outY, uint32_t width, uint32_t height );

        void _setup( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t widthIn, uint32_t heightIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                     uint32_t widthOut, uint32_t heightOut, bool oppositeAxis = false );

        void _setup( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2, Image & out, uint32_t startXOut,
                     uint32_t startYOut, uint32_t width, uint32_t height );

        void _setup( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out1, uint32_t startXOut1, uint32_t startYOut1, Image & out2, uint32_t startXOut2,
                     uint32_t startYOut2, Image & out3, uint32_t startXOut3, uint32_t startYOut3, uint32_t width, uint32_t height );

        void _setup( const Image & in1, uint32_t startXIn1, uint32_t startYIn1, const Image & in2, uint32_t startXIn2, uint32_t startYIn2, const Image & in3,
                     uint32_t startXIn3, uint32_t startYIn3, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height );

        virtual void _task( size_t taskId ) = 0;

        void _processTask(); // function which calls global thread pool and waits results from it

        void _validateTask();
    };
}
