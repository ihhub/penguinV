#pragma once

#include "../image_buffer.h"

// Memory bandwidth is a linear function f(x) = a * x + b
struct MemoryBandwidth
{
    MemoryBandwidth( double multiplier_ = 0.0, double addition_ = 0.0, size_t minimumSizeAddition_ = 1 )
        : multiplier         ( multiplier_          )
        , addition           ( addition_            )
        , minimumSizeAddition( minimumSizeAddition_ )
    {}

    double calculateTime( size_t memorySize )
    {
        if ( memorySize <= minimumSizeAddition )
            return addition;
        else
            return (multiplier * memorySize);
    }

    double multiplier, addition;
    size_t minimumSizeAddition;
};

class DeviceInfo
{
public:
    enum Type
    {
        CPU,
        CUDA,
        OPENCL
    };
};

struct TaskInfo
{
    
};

class TaskManager
{
public:
};
