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
#include "../performance_test_framework.h"
#include "../performance_test_helper.h"
#include "performance_test_image_function_opencl.h"
#include <iostream>

int main( int argc, char * argv[] )
{
    if ( !multiCL::isOpenCLSupported() ) {
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

        multiCL::MemoryManager::memory().reserve( 32 * 1024 * 1024 ); // preallocate memory (32 MB)

        PerformanceTestFramework framework;
        addTests_Image_Function_OpenCL( framework );

        framework.run();
    }

    return 0;
}
