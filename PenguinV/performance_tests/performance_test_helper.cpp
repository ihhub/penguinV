#include <algorithm>
#include <math.h>
#include <numeric>
#include <vector>
#include "performance_test_helper.h"
#include "../Library/thread_pool.h"
#include "../Library/image_exception.h"

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

            do {
                // The function removes 'biggest' element from an array is case
                // if a difference between a value and mean is more than 6 sigma
                mean  = sum / data.size();
                sigma = sqrt( (sumSquare / (data.size()) - mean * mean) * (data.size()) / (data.size() - 1) );

                double maximumValue = 0;
                std::vector < double >::const_iterator maximumPosition = data.begin();

                for( std::vector < double >::const_iterator v = data.begin(); v != data.end(); ++v ) {
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

            } while( removed || data.size() < 2 );
        }
    };
};

namespace Performance_Test
{
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

        _time.push_back( time.count() );
    }

    std::pair < double, double > TimerContainer::mean()
    {
        if( _time.empty() )
            throw imageException( "Cannot find mean value of empty array" );

        // We remove first value because it is on 'cold' cache
        _time.pop_front();

        // Remove all values what are out of +/- 6 sigma range
        std::vector < double > time ( _time.begin(), _time.end() );

        double mean = 0, sigma = 0;

        getDistribution( time, mean, sigma );

        // return results in milliseconds
        return std::pair<double, double>( 1000 * mean, 1000 * sigma );
    }

    Bitmap_Image::Image uniformImage( uint32_t width, uint32_t height )
    {
        return uniformImage( width, height, randomValue<uint8_t>( 256 ) );
    }

    Bitmap_Image::Image uniformImage( uint32_t width, uint32_t height, uint8_t value )
    {
        Bitmap_Image::Image image( width, height );

        image.fill( value );

        return image;
    }

    Bitmap_Image::Image uniformColorImage( uint32_t width, uint32_t height )
    {
        return uniformColorImage( width, height, randomValue<uint8_t>( 256 ) );
    }

    Bitmap_Image::Image uniformColorImage( uint32_t width, uint32_t height, uint8_t value )
    {
        Bitmap_Image::Image image( width, height, Bitmap_Image::RGB );

        image.fill( value );

        return image;
    }

    std::vector< Bitmap_Image::Image > uniformImages( uint32_t count, uint32_t width, uint32_t height )
    {
        std::vector < Bitmap_Image::Image > image( count );

        for( std::vector< Bitmap_Image::Image >::iterator im = image.begin(); im != image.end(); ++im )
            *im = uniformImage( width, height );

        return image;
    }

    uint32_t runCount()
    {
        return 1024;
    }

    void setFunctionPoolThreadCount()
    {
        Thread_Pool::ThreadPoolMonoid::instance().resize( 4 );
    }
}
