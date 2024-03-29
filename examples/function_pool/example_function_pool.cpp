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

// Example application of library's function pool utilization
#include "../../src/function_pool.h"
#include "../../src/image_function.h"
#include "../../src/penguinv_exception.h"
#include "../../src/thread_pool.h"
#include <cstdlib>
#include <iostream>

void basic( const std::vector<penguinV::Image> & frame );
void multithreaded( const std::vector<penguinV::Image> & frame );

double getElapsedTime( std::chrono::time_point<std::chrono::system_clock> start );

template <class T_>
T_ randomValue( uint32_t maximum )
{
    return static_cast<T_>( static_cast<uint32_t>( rand() ) % maximum );
}

int main()
{
    try // <---- do not forget to put your code into try.. catch block!
    {
        // This example is more technical and it shows the difference in software development using
        // basic functions and using functions with thread pool support.
        // Plus it evaluates the speed of calculations between two examples.
        // Please take a note that speed as well as thread count depends on your CPU and
        // overall system so do not think that this comparison is fully accurate

        // Conditions:
        // We have 60 big images (frames) with 1920 x 1080 pixels (60 fps fullHD :) )
        // Our aim is pretty simple: we have to find some suspicious objects on frames and highlight them
        // What we need to do is:
        // 1. get a difference between two neighbour frames
        // 2. Find optimum threshold value
        // 3. Compare threshold value with known value of background
        // 4. threshold image with threshold
        // 5. put result on the map

        // Create a set of 60 frames
        std::vector<penguinV::Image> frame( 60, penguinV::Image( 1920, 1080 ) );

        for ( std::vector<penguinV::Image>::iterator image = frame.begin(); image != frame.end(); ++image ) {
            // Fill background. Let's assume that background varies from 0 to 15 gray scale values
            image->fill( randomValue<uint8_t>( 16u ) );

            // Then add some 'suspicious' objects on some random images
            if ( randomValue<int>( 10 ) == 0 ) { // at least ~6 of 60 images would have objects
                // generate random place of object, in our case is rectangle
                const uint32_t x = randomValue<uint32_t>( image->width() * 2 / 3 );
                const uint32_t y = randomValue<uint32_t>( image->height() * 2 / 3 );
                const uint32_t width = randomValue<uint32_t>( image->width() - x );
                const uint32_t height = randomValue<uint32_t>( image->height() - y );

                // fill an area with some random value what is bigger than background value
                Image_Function::Fill( *image, x, y, width, height, static_cast<uint8_t>( randomValue<uint8_t>( 128u ) + 64u ) );
            }
        }

        std::cout << "----------" << std::endl << "Basic functions. Evaluating time..." << std::endl << "----------" << std::endl;
        std::chrono::time_point<std::chrono::system_clock> startTime = std::chrono::system_clock::now();

        basic( frame );

        std::cout << "Total time is " << getElapsedTime( startTime ) << " seconds" << std::endl;

        std::cout << "----------" << std::endl << "Functions with thread pool support. Evaluating time..." << std::endl << "----------" << std::endl;
        startTime = std::chrono::system_clock::now();

        multithreaded( frame );

        std::cout << "Total time is " << getElapsedTime( startTime ) << " seconds" << std::endl;
    }
    catch ( const std::exception & ex ) { // uh-oh, something went wrong!
        std::cout << ex.what() << ". Press any button to continue." << std::endl;
        std::cin.ignore();
        return 1;
    }
    catch ( ... ) { // uh-oh, something terrible happen!
        std::cout << "Generic exception raised. Press any button to continue." << std::endl;
        std::cin.ignore();
        return 2;
    }

    std::cout << "Application ended correctly." << std::endl;
    return 0;
}

double getElapsedTime( std::chrono::time_point<std::chrono::system_clock> start )
{
    const std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
    const std::chrono::duration<double> time = end - start;

    return time.count();
}

void basic( const std::vector<penguinV::Image> & frame )
{
    // Prepare image map
    penguinV::Image map( frame.front().width(), frame.front().height() );
    map.fill( 0 );

    penguinV::Image result( map.width(), map.height() );

    for ( size_t i = 0; i + 1 < frame.size(); ++i ) {
        // subtract one image from another
        Image_Function::Subtract( frame[i + 1], frame[i], result );

        // find optimal threshold
        uint8_t threshold = Image_Function::GetThreshold( Image_Function::Histogram( result ) );

        if ( threshold > 0 ) { // there is something more than just background!
            // threshold image
            Image_Function::Threshold( result, result, threshold );
            // add result to the map
            Image_Function::BitwiseOr( map, result, map );
        }
    }

    // here we have to save the image map but don't do this in the example
}

void multithreaded( const std::vector<penguinV::Image> & frame )
{
    // okay we setup 4 thread in global thread pool
    ThreadPoolMonoid::instance().resize( 4 );

    // Prepare image map
    penguinV::Image map( frame.front().width(), frame.front().height() );
    map.fill( 0 );

    penguinV::Image result( map.width(), map.height() );

    // As you see the only difference in this loop is namespace name
    for ( size_t i = 0; i + 1 < frame.size(); ++i ) {
        // subtract one image from another
        Function_Pool::Subtract( frame[i + 1], frame[i], result );

        // find optimal threshold
        uint8_t threshold = Image_Function::GetThreshold( Function_Pool::Histogram( result ) );

        if ( threshold > 0 ) { // there is something more than just background!
            // threshold image
            Function_Pool::Threshold( result, result, threshold );
            // add result to the map
            Function_Pool::BitwiseOr( map, result, map );
        }
    }

    // here we have to save the image map but don't do this in the example

    // We stop all threads in thread pool
    ThreadPoolMonoid::instance().stop();
}
