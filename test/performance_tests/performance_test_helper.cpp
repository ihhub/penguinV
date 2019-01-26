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

            const size_t sizeLimit = data.size() * 3 / 4; // remove maximum 25% of values

            do {
                // The loop removes 'biggest' element from an array in case if a difference between the value and mean is more than 6 sigma
                const uint32_t size = static_cast<uint32_t>( data.size() ); // reasonable as amount of input data cannot be so huge
                mean  = sum / size;
                sigma = sqrt( (sumSquare / size - mean * mean) * size / (size - 1) );

                double maximumValue = 0;
                std::vector < double >::iterator maximumPosition = data.begin();

                for( std::vector < double >::iterator v = data.begin(); v != data.end(); ++v ) {
                    const double value = fabs( (*v) - mean );

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
}
