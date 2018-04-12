#include <algorithm>
#include <math.h>
#include <numeric>
#include <vector>
#include "performance_test_helper.h"
#include "../../src/image_exception.h"

namespace
{
    // The function calculates mean and sigma value of distribution
    void getDistribution( std::vector < double > & data, double & mean, double & sigma )
    {
        if( data.empty() ) {
            mean = sigma = 0;
        }
        else if( data.size() == 1u ) {
            mean = data.front();
            sigma = 0;
        }
        else {
            double sum       = 0;
            double sumSquare = 0;

            for( std::vector < double >::const_iterator v = data.begin(); v != data.end(); ++v ) {
                sum += *v;
                sumSquare += (*v) * (*v);
            }

            bool removed = false;

            const size_t sizeLimit = data.size() * 3 / 4;

            do {
                // The function removes 'biggest' element from an array in case
                // if a difference between a value and mean is more than 6 sigma
                mean  = sum / data.size();
                sigma = sqrt( (sumSquare / (data.size()) - mean * mean) * (data.size()) / (data.size() - 1) );

                double maximumValue = 0;
                std::vector < double >::iterator maximumPosition = data.begin();

                for( std::vector < double >::iterator v = data.begin(); v != data.end(); ++v ) {
                    double value = fabs( (*v) - mean );

                    if( maximumValue < value ) {
                        maximumPosition = v;
                        maximumValue = value;
                    }
                }

                if( maximumValue > 6 * sigma ) {
                    sum = sum - (*maximumPosition);
                    sumSquare = sumSquare - (*maximumPosition) * (*maximumPosition);

                    data.erase( maximumPosition );
                    removed = true;
                }
                else {
                    removed = false;
                }
            } while( removed && (data.size() > sizeLimit) );
        }
    }

    PenguinV_Image::Image generateImage( uint32_t width, uint32_t height, uint8_t colorCount, uint8_t value )
    {
        PenguinV_Image::Image image( width, height, colorCount );

        image.fill( value );

        return image;
    }
}

namespace Performance_Test
{
    BaseTimerContainer::BaseTimerContainer()
    { }

    BaseTimerContainer::~BaseTimerContainer()
    { }

    std::pair < double, double > BaseTimerContainer::mean()
    {
        if( _time.empty() )
            throw imageException( "Cannot find mean value of empty array" );

        // We remove first value because it is on 'cold' cache
        _time.pop_front();

        // Remove all values what are out of +/- 6 sigma range
        std::vector < double > time ( _time.begin(), _time.end() );

        double mean  = 0.0;
        double sigma = 0.0;

        getDistribution( time, mean, sigma );

        // return results in milliseconds
        return std::pair<double, double>(  mean, sigma );
    }

    void BaseTimerContainer::push(double value)
    {
        _time.push_back( value );
    }


    TimerContainer::TimerContainer()
        : _startTime( std::chrono::high_resolution_clock::now() )
    { }

    TimerContainer::~TimerContainer()
    { }

    void TimerContainer::start()
    {
        _startTime = std::chrono::high_resolution_clock::now();
    }

    void TimerContainer::stop()
    {
        std::chrono::time_point < std::chrono::high_resolution_clock > endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration < double > time = endTime - _startTime;

        push( time.count() * 1000.0 ); // original value is in microseconds
    }

    std::pair < double, double > runPerformanceTest(performanceFunction function, uint32_t size )
    {
        Performance_Test::TimerContainer timer;
        function( timer, size  );
        return timer.mean();
    }

    PenguinV_Image::Image uniformImage( uint32_t width, uint32_t height )
    {
        return uniformImage( width, height, randomValue<uint8_t>( 256 ) );
    }

    PenguinV_Image::Image uniformImage( uint32_t width, uint32_t height, uint8_t value )
    {
        return generateImage( width, height, PenguinV_Image::GRAY_SCALE, value);
    }

    PenguinV_Image::Image uniformColorImage( uint32_t width, uint32_t height )
    {
        return uniformColorImage( width, height, randomValue<uint8_t>( 256 ) );
    }

    PenguinV_Image::Image uniformColorImage( uint32_t width, uint32_t height, uint8_t value )
    {
        return generateImage( width, height, PenguinV_Image::RGB, value);
    }

    std::vector< PenguinV_Image::Image > uniformImages( uint32_t count, uint32_t width, uint32_t height )
    {
        std::vector < PenguinV_Image::Image > image( count );

        for( std::vector< PenguinV_Image::Image >::iterator im = image.begin(); im != image.end(); ++im )
            *im = uniformImage( width, height );

        return image;
    }

    std::vector< PenguinV_Image::Image > uniformColorImages( uint32_t count, uint32_t width, uint32_t height )
    {
        std::vector < PenguinV_Image::Image > image( count );

        for( std::vector< PenguinV_Image::Image >::iterator im = image.begin(); im != image.end(); ++im )
            *im = uniformColorImage( width, height );

        return image;
    }

    uint32_t runCount()
    {
        return 1024;
    }
}
