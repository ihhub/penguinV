#include <iostream>
#include "unit_test_image_function_opencl.h"
#include "../unit_test_framework.h"
#include "../unit_test_helper.h"
#include "../../../src/opencl/opencl_device.h"
#include "../../../src/opencl/opencl_helper.h"

int main( int argc, char * argv[] )
{
    try 
    {
        // The main purpose of this application is to test everything within library
        // To do this we need an engine (framework) and a bunch of tests
        if ( !multiCL::isOpenCLSupported() ) {
            std::cout << "No GPU devices in the system" << std::endl;
            return 0;
        }

        Unit_Test::setRunCount( argc, argv, 1001 );

        multiCL::OpenCLDeviceManager & deviceManager = multiCL::OpenCLDeviceManager::instance();
        deviceManager.initializeDevices();

        int returnValue = 0;
        for ( uint32_t deviceId = 0; deviceId < deviceManager.deviceCount(); ++deviceId ) {
            deviceManager.setActiveDevice( deviceId );

            const multiCL::OpenCLDevice & device = deviceManager.device();
            std::cout << device.name() << ": " << device.computeCapability() << std::endl;

            multiCL::MemoryManager::memory().reserve( 32 * 1024 * 1024 ); // preallocate memory (32 MB)

            UnitTestFramework framework;
            addTests_Image_Function_OpenCL( framework );

            returnValue += framework.run();
        }

        return returnValue;
    } 
    catch ( const imageException & ex ) {
        std::cout << "Exception " << ex.what() << " raised during OpenCL unit tests." << std::endl;
        return 0;
    }
    catch ( ... ) {
        std::cout << "Unknown exception raised during OpenCL unit tests." << std::endl;
        return 0;
  }
}
