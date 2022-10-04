/***************************************************************************
 *   penguinV: https://github.com/ihhub/penguinV                           *
 *   Copyright (C) 2017 - 2022                                             *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

#include "../../../src/opencl/opencl_device.h"
#include "../../../src/opencl/opencl_helper.h"
#include "../unit_test_framework.h"
#include "../unit_test_helper.h"
#include "unit_test_image_function_opencl.h"
#include <iostream>

int main( int argc, char * argv[] )
{
    try {
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
    catch ( const penguinVException & ex ) {
        std::cout << "Exception " << ex.what() << " raised during OpenCL unit tests." << std::endl;
        return 0;
    }
    catch ( ... ) {
        std::cout << "Unknown exception raised during OpenCL unit tests." << std::endl;
        return 0;
    }
}
