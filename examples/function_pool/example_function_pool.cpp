// Example application of library's function pool utilization
#include <cstdlib>
#include <iostream>
#include "../../src/function_pool.h"
#include "../../src/image_exception.h"
#include "../../src/image_function.h"
#include "../../src/thread_pool.h"

void basic        ( const std::vector < PenguinV_Image::Image > & frame );
void multithreaded( const std::vector < PenguinV_Image::Image > & frame );

double getElapsedTime( std::chrono::time_point < std::chrono::system_clock > start );

template <class T_>
T_ randomValue( T_ maximum )
{
    return static_cast<T_>(rand()) % maximum;
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
        std::vector < PenguinV_Image::Image > frame( 60, PenguinV_Image::Image( 1920, 1080 ) );

        for( std::vector < PenguinV_Image::Image >::iterator image = frame.begin(); image != frame.end(); ++image ) {
            // Fill background. Let's assume that background varies from 0 to 15 gray scale values
            image->fill( randomValue<uint8_t>(16u) );

            // Then add some 'suspicious' objects on some random images
            if( randomValue<int>(10) == 0 ) { // at least ~6 of 60 images would have objects
                // generate random place of object, in our case is rectangle
                const uint32_t x      = randomValue<uint32_t>(image->width()  * 2 / 3);
                const uint32_t y      = randomValue<uint32_t>(image->height() * 2 / 3);
                const uint32_t width  = randomValue<uint32_t>(image->width()  - x);
                const uint32_t height = randomValue<uint32_t>(image->height() - y);

                // fill an area with some random value what is bigger than background value
                Image_Function::Fill( *image, x, y, width, height, randomValue<uint8_t>(128u) + 64u );
            }
        }

        std::cout << "----------" << std::endl
            << "Basic functions. Evaluating time..." << std::endl << "----------" << std::endl;
        std::chrono::time_point < std::chrono::system_clock > startTime = std::chrono::system_clock::now();

        basic( frame );

        std::cout << "Total time is " << getElapsedTime( startTime ) << " seconds" << std::endl;

        std::cout << "----------" << std::endl
            << "Functions with thread pool support. Evaluating time..." << std::endl << "----------" << std::endl;
        startTime = std::chrono::system_clock::now();

        multithreaded( frame );

        std::cout << "Total time is " << getElapsedTime( startTime ) << " seconds" << std::endl;
    }
    catch( const imageException & ex ) {
        // uh-oh, something went wrong!
        std::cout << "Exception " << ex.what() << " raised. Do your black magic to recover..." << std::endl;
        // your magic code must be here to recover from bad things
        return 1;
    }
    catch( ... ) {
        // uh-oh, something really terrible happen!
        std::cout << "Something really terrible happen. No idea what it is. Do your black magic to recover..." << std::endl;
        // your magic code must be here to recover from terrible things
        return 2;
    }

    std::cout << "Everything went fine." << std::endl;

    return 0;
}

double getElapsedTime( std::chrono::time_point < std::chrono::system_clock > start )
{
    const std::chrono::time_point < std::chrono::system_clock > end = std::chrono::system_clock::now();
    const std::chrono::duration < double > time = end - start;

    return time.count();
}

void basic( const std::vector < PenguinV_Image::Image > & frame )
{
    // Prepare image map
    PenguinV_Image::Image map( frame.front().width(), frame.front().height() );
    map.fill( 0 );

    PenguinV_Image::Image result( map.width(), map.height() );

    for( size_t i = 0; i + 1 < frame.size(); ++i ) {
        // subtract one image from another
        Image_Function::Subtract( frame[i + 1], frame[i], result );

        // find optimal threshold
        uint8_t threshold = Image_Function::GetThreshold( Image_Function::Histogram( result ) );

        if( threshold > 0 ) { // there is something more than just background!
            // threshold image
            Image_Function::Threshold( result, result, threshold );
            // add result to the map
            Image_Function::BitwiseOr( map, result, map );
        }
    }

    // here we have to save the image map but don't do this in the example
}

void multithreaded( const std::vector < PenguinV_Image::Image > & frame )
{
    // okay we setup 4 thread in global thread pool
    Thread_Pool::ThreadPoolMonoid::instance().resize( 4 );

    // Prepare image map
    PenguinV_Image::Image map( frame.front().width(), frame.front().height() );
    map.fill( 0 );

    PenguinV_Image::Image result( map.width(), map.height() );

    // As you see the only difference in this loop is namespace name
    for( size_t i = 0; i + 1 < frame.size(); ++i ) {
        // subtract one image from another
        Function_Pool::Subtract( frame[i + 1], frame[i], result );

        // find optimal threshold
        uint8_t threshold = Image_Function::GetThreshold( Function_Pool::Histogram( result ) );

        if( threshold > 0 ) { // there is something more than just background!
            // threshold image
            Function_Pool::Threshold( result, result, threshold );
            // add result to the map
            Function_Pool::BitwiseOr( map, result, map );
        }
    }

    // here we have to save the image map but don't do this in the example

    // We stop all threads in thread pool
    Thread_Pool::ThreadPoolMonoid::instance().stop();
}
