#pragma once

#include <chrono>
#include <cstdlib>
#include <list>
#include <vector>
#include "../../src/image_buffer.h"
#include "../test_helper.h"

namespace Performance_Test
{

  using namespace Test_Helper;
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

    typedef void(*performanceFunction)( TimerContainer &, uint32_t);
    std::pair < double, double > runPerformanceTest(performanceFunction function, uint32_t size );

    uint32_t runCount(); // fixed value for all test loops

}
