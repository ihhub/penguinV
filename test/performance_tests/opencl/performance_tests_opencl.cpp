// This application is designed to run performance tests on penguinV library
#include <iostream>
#include "performance_test_image_function_opencl.h"
#include "../performance_test_framework.h"
#include "../performance_test_helper.h"
#include "../../../src/opencl/opencl_device.h"
#include "../../../src/opencl/opencl_helper.h"

int main( int argc, char * argv[] )
{
    // The main purpose of this application is to test everything within library
    // To do this we need an engine (framework) and a bunch of tests
    if( !multiCL::isOpenCLSupported() ) {
        std::cout << "No GPU devices in the system" << std::endl;
        return 0;
    }

    Performance_Test::setRunCount( argc, argv, 128 );

    multiCL::OpenCLDeviceManager & deviceManager = multiCL::OpenCLDeviceManager::instance();
    deviceManager.initializeDevices();

    for ( uint32_t deviceId = 0; deviceId < deviceManager.deviceCount(); ++deviceId ) {
        deviceManager.setActiveDevice( deviceId );

        const multiCL::OpenCLDevice & device = deviceManager.device();
        std::cout << device.name() << ": " << device.computeCapability() << std::endl;

        // We preallocate memory (32 MB)
        multiCL::MemoryManager::memory().reserve( 32 * 1024 * 1024 );

        // We create a framework and add tests
        PerformanceTestFramework framework;
        addTests_Image_Function_OpenCL( framework );

        // Just run the framework what will handle all tests
        framework.run();
    }

    return 0;
}
