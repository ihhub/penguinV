#pragma once

#include <chrono>
#include <cstdlib>
#include <list>
#include <vector>
#include "../../src/image_buffer.h"

namespace Performance_Test
{
    // A class to measure time of individual test
    class BaseTimerContainer
    {
    public:
        BaseTimerContainer();
        ~BaseTimerContainer();

        std::pair < double, double > mean(); // returns mean and sigma values
    protected:
        void push(double value);
    private:
        std::list < double > _time;
    };

    class TimerContainer : public BaseTimerContainer
    {
    public:
        TimerContainer();
        ~TimerContainer();

        void start(); // start time measurement
        void stop();  // stop time measurement
    private:
        std::chrono::time_point < std::chrono::high_resolution_clock > _startTime;
    };

    // Functions to generate images
    Bitmap_Image::Image uniformImage( uint32_t width, uint32_t height );
    Bitmap_Image::Image uniformImage( uint32_t width, uint32_t height, uint8_t value );
    Bitmap_Image::Image uniformColorImage( uint32_t width, uint32_t height );
    Bitmap_Image::Image uniformColorImage( uint32_t width, uint32_t height, uint8_t value );
    std::vector< Bitmap_Image::Image > uniformImages( uint32_t count, uint32_t width, uint32_t height );

    uint32_t runCount(); // fixed value for all test loops

    void setFunctionPoolThreadCount(); // by default the value is 4 you can change it to make better results

    // Return random value for specific range or variable type
    template <typename data>
    data randomValue( int maximum )
    {
        if( maximum <= 0 )
            return 0;
        else
            return static_cast<data>(rand() % maximum);
    }

    template <typename data>
    data randomValue( data minimum, int maximum )
    {
        if( maximum <= 0 ) {
            return 0;
        }
        else {
            data value = static_cast<data>(rand() % maximum);

            if( value < minimum )
                value = minimum;

            return value;
        }
    }
}
