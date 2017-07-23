// This application is designed to run unit tests on penguinV library
#include <iostream>
#include "unit_test_image_function_cuda.h"
#include "../unit_test_framework.h"
#include "../../Library/cuda/cuda_memory.cuh"

int main()
{
    // The main purpose of this application is to test everything within library
    // To do this we need an engine (framework) and a bunch of tests

    // We preallocate memory (32 MB)
    Cuda_Memory::MemoryAllocator::instance().reserve( 32 * 1024 * 1024 );

    // We create a framework
    Unit_Test::UnitTestFramework framework;

    // We add tests
    Unit_Test::addTests_Image_Function_Cuda( framework );

    // Just run the framework what will handle all tests
    return framework.run();
}
